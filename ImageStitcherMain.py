import cv2
import os
import Stitcher as st
import imutils

fileDirectory = "homework_dataset\\data_image_stitching"
files = [i for i in os.listdir(fileDirectory) if i.endswith('.jpg') or i.endswith('.png')]
print(files)

sticher = st.Stitcher_class()


for i in files: 
    currentfile = os.path.join(fileDirectory,i)
    print( currentfile)
    img = cv2.imread(currentfile)
    (kp, f) = sticher.detectAndDescribe(img)

imageA2 = imutils.resize( cv2.imread(os.path.join(fileDirectory,files[0])), width=400)
imageB2 = imutils.resize(cv2.imread(os.path.join(fileDirectory,files[1])), width=400)
 

imageA = imutils.resize( cv2.imread(os.path.join(fileDirectory,files[2])), width=400)
imageB = imutils.resize(cv2.imread(os.path.join(fileDirectory,files[3])), width=400)
imageC = imutils.resize(cv2.imread(os.path.join(fileDirectory,files[4])), width=400)
imageD = imutils.resize(cv2.imread(os.path.join(fileDirectory,files[5])), width=400)


imageA3 = imutils.resize( cv2.imread(os.path.join(fileDirectory,files[6])), width=400)
imageB3 = imutils.resize(cv2.imread(os.path.join(fileDirectory,files[7])), width=400)


# stitch the images together to create a panorama
result = sticher.stitch(imageA2,imageB2, MAX_ITER = 60000)

cv2.imshow("Result1", result)
cv2.waitKey(0)
exit(0)

result2 = sticher.stitch4images(imageA,imageB,imageC,imageD , MAX_ITER = 160000)
cv2.imshow("Result2", result2)
cv2.waitKey(0)
exit(0)

result3 =  sticher.stitch(imageA3,imageB3, MAX_ITER = 40000)
cv2.imshow("Result3", result3)
cv2.waitKey(0)
exit(0)
