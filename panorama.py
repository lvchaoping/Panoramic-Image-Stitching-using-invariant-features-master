import numpy as np
import imutils
import cv2
import copy
class Panorama:#类：全景
    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=4.0,match_status=False):
        #从SIFT中检测特征和关键点
        (imageB, imageA) = images#为什么不是A,B？答：方便后面拼接
        (KeypointsA, features_of_A) = self.Detect_Feature_And_KeyPoints(imageA)
        (KeypointsB, features_of_B) = self.Detect_Feature_And_KeyPoints(imageB)
        #得到有效的匹配点
        Values = self.matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)
        if Values is None:
            return None
        #使用计算好的单应性H获得图像A的透视（实际上是B，可能方便直接将A放进左边的画布上）
        (matches, Homography, status) = Values
      #  print(Homography)
     #   getchar()
        result_image = self.getwarp_perspective(imageA,imageB,Homography)
        trans=copy.deepcopy(result_image)
       # cv2.imwrite("trans.jpg",result_image)
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # 检查匹配的关键点是否应该可视化
        (left_top_x,left_bottom_x)=self.CalcCorners(Homography,imageA)
        result_image=self.OptimizeSeam(imageB,trans,result_image,left_top_x,left_bottom_x)
        if match_status:
            vis = self.draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches,status)
            return (result_image, vis)
        return result_image
    def getwarp_perspective(self,imageA,imageB,Homography):
        val = imageA.shape[1] + imageB.shape[1]
        result_image = cv2.warpPerspective(imageA, Homography, (val , imageA.shape[0]))
        #对图A进行透视H变换到B的视角，输出形状是拼接的宽和高？ 
        return result_image
    def Detect_Feature_And_KeyPoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#这句好像没用到
        # 从图像中检测和提取特征
        descriptors = cv2.xfeatures2d.SIFT_create()
        (Keypoints, features) = descriptors.detectAndCompute(image, None)
        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, features)
    def get_Allpossible_Match(self,featuresA,featuresB):
        # 用欧式距离计算所有匹配点的距离 并且opencv为此提供了
        #DescriptorMatcher_create() 函数
        match_instance = cv2.DescriptorMatcher_create("BruteForce")#建立暴力匹配器
        All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)#为A的每个特征点找出两个最佳匹配点？
        return All_Matches
    def All_validmatches(self,AllMatches,lowe_ratio):
        #All_Matches中的两个特征点进行比较，低于比率的作为有效的匹配点
        valid_matches = []
        for val in AllMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))#匹配的两个特征点的索引加进来
        return valid_matches
    def Compute_Homography(self,pointsA,pointsB,max_Threshold):
        #用两个图像中的点作单映性变换
        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        #max_Threshold：允许将点对视为内点的最大阈值。
        return (H,status)
    def matchKeypoints(self, KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):
        AllMatches = self.get_Allpossible_Match(featuresA,featuresB);
        valid_matches = self.All_validmatches(AllMatches,lowe_ratio)
        if len(valid_matches) > 4:
            # construct 两个点集
            pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
            pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])
            (Homograpgy, status) = self.Compute_Homography(pointsA, pointsB, max_Threshold)
            return (valid_matches, Homograpgy, status)
        else:
            return None
    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]#：2表示从开始到第二个参数
        return (h,w)
    def get_points(self,imageA,imageB):#返回放到一起的矩阵
        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        #创建全0 矩阵，高取高的一个，宽相加
        vis[0:hA, 0:wA] = imageA#图片的放置
        vis[0:hB, wA:] = imageB
        return vis
    def draw_Matches(self, imageA, imageB, KeypointsA, KeypointsB, matches, status):
        (hA,wA) = self.get_image_dimension(imageA)
        vis = self.get_points(imageA,imageB)
        # 用于匹配点的循环
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:#估计状态只有0/1?
                ptA = (int(KeypointsA[queryIdx][0]), int(KeypointsA[queryIdx][1]))
                ptB = (int(KeypointsB[trainIdx][0]) + wA, int(KeypointsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)#匹配点间画线
        return vis
    def CalcCorners( self,H, src):
        V2 =np.array([[0.0], [0.0],[1.0]]) #左上角, V1变换后的坐标值#列向量
        V2=np.mat(V2)
       # v1 =np.array([[0.0], [0.0], [1.0]]) 
       # V2=np.transpose(v2)
        H=np.mat(H)
        V1=H*V2#左上角(0,0,1)
        left_top_x=V1[0][0]/V1[2][0]#左上角(0,src.rows,1)
        V2[0][0] = 0 
        V2[1][0] = src.shape[0]
        V2[2][0] = 1 
      #  V2=np.transpose(v2)#列向量
      #  H=np.mat(H)
        V1 = H * V2 
        left_bottom_x = V1[0][0] / V1[2][0]
        return left_top_x,left_bottom_x
    def OptimizeSeam(self,img1,trans,dst,left_top_x, left_bottom_x):
        start = int(min(left_top_x, left_bottom_x))#开始位置，即重叠区域的左边界  
        processWidth = float(img1.shape[1] - start)#重叠区域的宽度  
        rows = int(dst.shape[0])
        cols = int(img1.shape[1]) #注意，是列数*通道数
        alpha = 1.0#img1中像素的权重  
        i=0
        j = start
        for i in range(rows):
            p = img1[i]  #获取第i行的首地址
            t = trans[i]
            d = dst[i]
            for j in range(cols):
                #如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
                if (t[j][0] == 0 and t[j][1] == 0 and t[j][2] == 0):
                    alpha = 1
                else:
                    #img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
                    alpha = (processWidth - (j - start)) / processWidth
                dst[i][j][0] = p[j][0] * alpha + t[j][0] * (1 - alpha)
                dst[i][j][1] = p[j][1] * alpha + t[j][1] * (1 - alpha)
                dst[i][j][2] = p[j][2] * alpha + t[j][2] * (1 - alpha)
      #  cv2.waitKey(0)
        return dst
    
