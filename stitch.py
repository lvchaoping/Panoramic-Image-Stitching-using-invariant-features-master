from panorama import Panorama
import imutils
import cv2
#从文件夹中载入图片: 1Hill & 2Hill, S1 & S2, 1 & 2
imageA = cv2.imread('.\Py\Panoramic-Image-Stitching-using-invariant-features-master\\5.JPG')
imageB = cv2.imread('.\Py\Panoramic-Image-Stitching-using-invariant-features-master\\6.JPG')
imageA = imutils.resize(imageA, width=400)#宽度调整为400，高度会跟着自适应调整(400*250)
imageB = imutils.resize(imageB, width=400)
panorama = Panorama()
(result, matched_points) = panorama.image_stitch([imageA, imageB], match_status=True)
#显示获得的全景图像和有效匹配点
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", matched_points)
cv2.imshow("Result", result)
#写入图片
cv2.imwrite(".\Py\Panoramic-Image-Stitching-using-invariant-features-master\\Matched_points.jpg",matched_points)#注意.jpg不能少
cv2.imwrite(".\Py\Panoramic-Image-Stitching-using-invariant-features-master\\Panorama.jpg",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
