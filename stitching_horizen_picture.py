from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2

root = Tk()

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img1 = cv2.imread(path)

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img2 = cv2.imread(path)

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img3 = cv2.imread(path)

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img4 = cv2.imread(path)

root.withdraw()

MIN_MATCH_COUNT = 4
FLANN_INDEX_KDTREE = 0

def stitchImgs(img2, img1):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches :
        if m.distance < 0.83*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
              
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
        h, w, channel = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
                
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

    width = img2.shape[1] + img1.shape[1]
    height = img2.shape[0] + img1.shape[0]
                   
    dst = cv2.warpPerspective(img1, M, (width,height))
                    
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    return dst            

def stitchQuad(img1, img2, img3, img4):
    img1=cv2.flip(img1,1)
    img2=cv2.flip(img2,1)
    
    halfLeft = stitchImgs(img2, img1)
    halfLeft = cv2.flip(halfLeft,1)
    halfRight = stitchImgs(img3, img4)
    
    return stitchImgs(halfLeft,halfRight)

resultImg = stitchQuad(img1, img2, img3, img4)

cv2.imshow("Stitching",resultImg)
                 
cv2.waitKey()
cv2.destroyAllWindows()
                    
