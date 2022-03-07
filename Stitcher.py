
import cv2
import numpy as np
import imutils
import math
import os
import imutils
import random
from PIL import Image

class Stitcher_class:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)
        # print("initilazed a stitcher")

    def stitch4images(self,image1,image2,image3,image4, ratio = 0.75, threshold = 4.0):
        
        result1= self.stitch(image2, image3,ratio,threshold)
        flippedresult1 = cv2.flip(result1, 1)
        # cv2.imshow("Result", result1)
        # cv2.waitKey(0)
        result2 = self.stitch(image1,image2,ratio,threshold)
        flippedresult2 = cv2.flip(result2, 1)
        # cv2.imshow("Result2", result2)
        # cv2.waitKey(0)
        result3 = self.stitch(flippedresult1,flippedresult2, ratio,threshold, MAX_ITER = 50000),
        flippedresult3= cv2.flip(result3, 1)
        # cv2.imshow("Result3", result3)
        # cv2.waitKey(0)
        result4 = self.stitch(image3,image4,ratio,threshold)
        flippedresult4 = cv2.flip(result4,1)
        # cv2.imshow("Result4", result4)
        # cv2.waitKey(0)
        result5 = self.stitch(flippedresult4,flippedresult3,ratio,threshold, MAX_ITER = 50000)
        # cv2.imshow("Result5", result5)
        # cv2.waitKey(0)
        result5 = cv2.flip(result5,1)
        return (result5)


    def stitch(self,image2, image1, ratio = 0.75, threshold = 4.0, MAX_ITER = 50000):
        print("Start SIFT")
        key1, desc1 = self.detectAndDescribe(image1)
        key2, desc2 = self.detectAndDescribe(image2)
        print("End SIFT")
        print("Start matching key points")
        model = self.matchKeypoints(key1, key2,desc1,desc2,ratio,threshold, MAX_ITER)
        print("End matching key points")
        # if the match is None, then there aren't enough matchedkeypoints to create a panorama
        if model is None:
            return None
        (matches, H,status) = model
        print("Start warping points")

        result,alpha = self.warpPerspective(image1, H, (image1.shape[0] , image1.shape[1]+ image2.shape[1]))
        print(image2.shape,result.shape)
        # result = self.alpha_blend(result,image2,alpha)
        result[0:image2.shape[0], 0:image2.shape[1],:] = image2
        print("End warping points")
        return result

    # detect and extract features from the image
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features



    def matchKeypoints(self,kpsA, kpsB, features1, features2, ratio, threshold, MAX_ITER = 50000):
        # compute the raw matches and initialize the list of actual matches
        distances,rawMatches = self.matchFeatures(features1,features2)

        matches = []

        for i in range (len(rawMatches)):
            # Lowe's ratio test
            if distances[i][0] < distances[i][1] * ratio :
                matches.append((rawMatches[i][0],i))
		# computing a homography requires at least 4 matches
        print("Number of matches found is : " + str(len(matches)))

        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            print("computing homography")
            #Own implemented homography
            (H,status) = self.findHomography(ptsA, ptsB, MAX_ITER)
            print(H)
            return (matches, H, status)
        return None


    def matchFeatures(self, features1, features2, k = 2):
        # print("feature shape  is " + str(features1.shape))
        disc=[]
        indx = []
        for i,feat1 in enumerate(features1):
            distances = []
            for j,feat2 in enumerate(features2):
                #find the mean of the features distances
                distances.append((np.linalg.norm(feat1-feat2),j))
            # Smallest K elements indices using sorted() + lambda + list slicing 
            distances.sort()
            disc.append([d for d,x in distances[:k]])
            indx.append([y for d,y in distances[:k]])
            # print('calculating the descriptor = {} is done'.format(i))

        return (disc,indx)
       
    def findHomography(self,ptsA,ptsB,MAXITER = 90000):
        
        #Initilaze the variables
        H = np.ones([9,1])
        points1 = ptsA
        points2 = ptsB
        bestH = np.ones([9,1])
        ransacsize = 4 #How many points we take for the subset
        MSE = 999999999 #keep track of minumum error ( LEAST SQUARED ERROR)

        #RANSAC runs MAXITER times. 
        for i in range (MAXITER):

            #Report the current minimal error to track if we are doing better solutions
            if (i % 1000) == 0:
                print(str(i)+"th iteration")
                print("MSE is currently min : " +str(MSE)) 
            homo = []
            h = np.ones([9,1])
            rn = np.random.randint(len(points1), size=ransacsize)
            A = points1[rn,:]
            B = points2[rn,:]

            # Get and Normalize coordinates:
            x1 = A[:,0]
            y1 = A[:,1]
            x2 = B[:,0]
            y2 = B[:,1]
            
            # Self Construction
            # Find the homography matrix between A and B.
            for i in range(len(A)):
                homo.append([x1[i], y1[i], 1, 0, 0, 0, (-1*x1[i]*x2[i]), (-1*y1[i]*x2[i]), (-1*x2[i])])
                homo.append([0, 0, 0, x1[i], y1[i], 1, (-1*x1[i]*y2[i]), (-1*y1[i]*y2[i]), (-1*y2[i])])
            
            #Construct one homography matrix by using singular value construction
            _,_,V = np.linalg.svd(homo)
            h = V[-1,:]
            #Normalize the homography matrix so that the last element is one.
            h = [x/h[8] for x  in h ] 
            H = np.reshape(h, (3,3))

            #Start testing the homography matrix
            error = 0
            # Test is for only half of the points. Because we may have outliers which would cause wrong
            # matchings if they increase the error.
            rn2 = np.random.randint(len(points1), size=(int(len(points1)/3)))
            test_p1= points1[rn2,:]
            test_p2 = points2[rn2,:]
            for hi in range(len(test_p1)):
                x = test_p1[hi,0]
                y = test_p1[hi,1]
                
                #Calculate the transformed points with homography matrix
                denum = ((H[2][0] * x )+ (H[2][1] * y) + 1)
                x_ = (H[0][0] * x + H[0][1] * y + H[0][2]) / denum
                y_ = (H[1][0] * x + H[1][1] * y + H[1][2]) / denum
                #Least squared errors, to decide if we are generalizing the homography for the data.
                error += (test_p2[hi,0]-x_)**2 + (test_p2[hi,1]-y_)**2
                
            if error < MSE: 
                MSE = error
                bestH = H

        return bestH, True


    #  TODO : fix this. 
    def warpPerspective(self,src_image,H,distance):
        height,width = distance
        # Bottom right
        x = src_image.shape[1] -1
        y = src_image.shape[0] - 1
        denum = (H[2][0] * x )+ (H[2][1] * y) + 1
        x_max1 = math.ceil( (H[0][0] * x + H[0][1] * y + H[0][2])/ denum )
        y_max1 = math.ceil( (H[1][0] * x + H[1][1] * y + H[1][2])/ denum )

        # Top right
        y = 0
        denum = (H[2][0] * x )+ (H[2][1] * y) + 1
        x_max2 = math.ceil( (H[0][0] * x + H[0][1] * y + H[0][2])/ denum )
        y_max2 = math.ceil( (H[1][0] * x + H[1][1] * y + H[1][2])/ denum )

        # Get max
        x_max = max(x_max1,x_max2)
        y_max = max(y_max1,y_max2)

        print("Distance input is "+ str (distance))
        result = np.zeros((height,width,3), np.uint8)
        alpha = np.ones((height,width,3), np.uint8)

        for y in range(src_image.shape[0]):
            for x in range(src_image.shape[1]):
                denum = (H[2][0] * x )+ (H[2][1] * y) + 1
                # TODO : fix possible lose of information

                x_1 = math.floor( (H[0][0] * x + H[0][1] * y + H[0][2])/ denum )
                y_1 = math.floor( (H[1][0] * x + H[1][1] * y + H[1][2])/ denum )
                if 0 <= y_1 < height and  0 <= x_1 < width:
                    result[y_1,x_1] = src_image[y,x,:]
                            
        for y in range(result.shape[0]-1):
            for x in range(result.shape[1]-1):
                if(all(v==0 for v in  result[y,x,:])):
                    alpha[y,x] =  [255,255,255]
                else:
                    alpha[y,x] = [0,0,0]

                    
        for y in range(src_image.shape[0]):
            for x in range(src_image.shape[1]):
                denum = (H[2][0] * x )+ (H[2][1] * y) + 1
                # TODO : fix possible lose of information

                # y_h,x_h,_ = H_inv @ np.array([y, x, 1])
                x_1 = math.floor( (H[0][0] * x + H[0][1] * y + H[0][2])/ denum )
                y_1 = math.floor( (H[1][0] * x + H[1][1] * y + H[1][2])/ denum )
                x_2 = math.ceil( (H[0][0] * x + H[0][1] * y + H[0][2])/ denum )
                y_2 = math.ceil( (H[1][0] * x + H[1][1] * y + H[1][2])/ denum )
                if 0 <= y_1 < height and 0 <= x_1 < width and 0 <=  y_2 < height and 0 <= x_2 < width:
                    if x_1 == x_2 and y_1 == y_2:
                        result[y_1,x_1] = src_image[y,x,:]
                    elif x_1 < x_2 and y_1 >= y_2: 
                        result[y_2,x_2] = src_image[y,x,:]
                        result[y_2,x_1] = src_image[y,x,:]
                        result[y_1,x_2] = src_image[y,x,:]
                    elif x_1 >= x_2 and  y_1 < y_2:
                        result[y_2,x_2] = src_image[y,x,:]
                        result[y_1,x_2] = src_image[y,x,:]
                        result[y_2,x_1] = src_image[y,x,:]
                    else:
                        result[y_1,x_2] = src_image[y,x,:]
                        result[y_2,x_1] = src_image[y,x,:]
                        result[y_2,x_2] = src_image[y,x,:]
                        result[y_1,x_1] = src_image[y,x,:]
        

        

        alpha = cv2.blur(alpha,(50,50))
        # cv2.imshow("ALPHA ", alpha)
        # cv2.waitKey(0)
        return result,alpha

   
   
    def alphaBlend(self,img1, img2, mask):

        
        im1_extended = np.zeros((img2.shape[0],img2.shape[1],3), np.uint8)
        im1_extended[0:img1.shape[0], 0:img1.shape[1],:] = img1
        

        cv2.imshow("img_2 " , im1_extended )
        cv2.waitKey(0)
        cv2.imshow("img_1 " , img2 )
        cv2.waitKey(0)
        """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
        """
        if mask.ndim==3 and mask.shape[-1] == 3:
            alpha = mask/255.0
        else:
            alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
        blended = cv2.convertScaleAbs(*(1-alpha) + img2*alpha)
        return blended

    def alpha_blend(self,im1,im2,alpha):
        
        print(im1.shape)
        print(im2.shape)

        def ispixelblack(pixel):
            return all(color==0 for color in pixel)

        def returnfirstpixel(row):
            for i,pixel in enumerate(row):
                if not ispixelblack(pixel):
                    return i
            return 0
        
        x_first = im1.shape[1]
        for row in im1:
            temp = returnfirstpixel(row)
            if temp < x_first:
                x_first = temp
        x_last = im2.shape[1]
        w_inter = x_last - x_first
        print("blend between:",x_first,x_last)

        intersection_mask = np.zeros(im1.shape)
        for y in range(intersection_mask.shape[0]):
            for x in range(x_first,x_last):
                p1, p2 = im1[y,x,:], im2[y,x,:]
                if not ispixelblack(p1) and not ispixelblack(p2): #intersection
                    intersection_mask[y,x,:] = [1,1,1]
        
        BLUR_SIZE = (im1.shape[0],im1.shape[1])
        grad1 = intersection_mask.copy().astype(float)
        grad1[x_first+w_inter:x_last,:,:] = [0,0,0]
        grad1 = cv2.blur(grad1,BLUR_SIZE)
        result1 = cv2.multiply(grad1, im1.astype(float))

        BLUR_SIZE = (im2.shape[0],im2.shape[1])
        grad2 = intersection_mask.copy().astype(float)[:im2.shape[0],:im2.shape[1],:]
        grad2[x_first:x_first+w_inter,:,:] = [0,0,0]
        grad2 = cv2.blur(grad2,BLUR_SIZE)
        result2 = np.zeros(im1.shape,np.uint8)
        result2[:im2.shape[0],:im2.shape[1],:] = cv2.multiply(grad2, im2.astype(float))

        blend = np.add(result1,result2).astype(np.uint8)
        blend_inter = np.multiply(blend, intersection_mask).astype(np.uint8)

        result = np.add(im1,blend_inter)
        result = np.add(result,result2).astype(np.uint8)


        # cv2.imshow("im1 ", im1)
        # cv2.imshow("im2 ", im2)
        # cv2.imshow("result1 ", result1)
        # cv2.imshow("result2 ", result2)
        # cv2.imshow("blend ", blend)
        # cv2.imshow("blend_inter ", blend_inter)
        # cv2.imshow("result ", result)
        # cv2.waitKey(0)
        # exit()

        return result

 # def alpha_blend(self,im1,im2,alpha):
    #     # Convert uint8 to float
    #     foreground = im1.astype(float)
    #     tem = np.zeros((im1.shape[0],im1.shape[1],3), np.uint8)
        
    #     tem[0:im2.shape[0], 0:im2.shape[1],:] = im2

    #     background = tem.astype(float)
    #     # Normalize the alpha mask to keep intensity between 0 and 1

    #     alpha =alpha.astype(float)/255
        
    #     # Multiply the foreground with the alpha matte
    #     foreground = cv2.multiply(alpha, foreground)

    #     # Multiply the background with ( 1 - alpha )
    #     background = cv2.multiply(1.0 - alpha, background)

    #     # Add the masked foreground and background.
    #     outImage = cv2.add(foreground, background)  

    #     return outImage/255

