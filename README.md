# Panoramic-Image-Stitching-using-invariant-features

I have created Panoramic image using invariant feature. I have basically imlemented the David Lowe paper on Automatic Image stitching using Invariants feature
After running the code you will find the images like in panaroma_image.jpg of two images 1.jpg and 2.jpg. Matched_points is showing the valid matches in images. You can also test on other samples attached or on your own images.
为了运行代码，你应该运行程序stitch.py。使用SIFT检测特征，然后使用RANSAC，计算Homography和匹配点以及warp预定以获得最终的全景图像。

环境：win10 + VScode python 3.7.1 + opencv4

