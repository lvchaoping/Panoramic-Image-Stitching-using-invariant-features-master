# Panoramic-Image-Stitching-using-invariant-features

I have created Panoramic image using invariant feature. I have basically imlemented the David Lowe paper on Automatic Image stitching using Invariants feature
After running the code you will find the images like in panaroma_image.jpg of two images 1.jpg and 2.jpg. Matched_points is showing the valid matches in images. You can also test on other samples attached or on your own images.
In order to run the code you should run the program stitch.py. Used SIFT to detect feature and then RANSAC, compute Homography and matched points and warp prespective to get final panoramic image.

environment: win10+VScode python 3.7.1 +opencv4


