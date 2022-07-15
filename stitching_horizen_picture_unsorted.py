from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2

class imgNode(object):
    def __init__(self, img, minCom = 0, next = None):
        self.img = img
        self.minCom = minCom
        self.next = next

root = Tk()

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img1 = cv2.imread(path)
fst = imgNode(img1)

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img2 = cv2.imread(path)
snd = imgNode(img2)

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img3 = cv2.imread(path)
trd = imgNode(img3)

path = filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img4 = cv2.imread(path)
fth = imgNode(img4)

root.withdraw()

beforeSort = [fst, snd, trd, fth]

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
        return None

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

def compareImg(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
        
    flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    matches = flann.knnMatch(des1,des2,k=2)
        
    p1=0
    p2=0
    cnt=0
        
    good = []
    for m,n in matches :
        if m.distance < 0.90*n.distance:
            good.append(m)
            p1 += kp1[m.queryIdx].pt[0]
            p2 += kp2[m.trainIdx].pt[0]
            cnt += 1
    
    if len(good)<MIN_MATCH_COUNT:
        return 0

    h,w,channel = img1.shape    
    p1 = p1/cnt/w*100
    h,w,channel = img2.shape
    p2 = p2/cnt/w*100

    return p1-p2

headIndex=0

for i in range(0,3):
    for j in range(i+1,4):
        dec = compareImg(beforeSort[i].img, beforeSort[j].img)
        if dec>10:
            if beforeSort[i].minCom < dec:
                beforeSort[i].next = beforeSort[j]
                beforeSort[i].minCom = dec
        elif dec<-10:
            dec *= -1
            if beforeSort[j].minCom < dec:
                beforeSort[j].next = beforeSort[i]
                beforeSort[j].minCom = dec
                
for i in range(0,4):
    if beforeSort[i].next != None:
        if beforeSort[i].next.next != None:
            if beforeSort[i].next.next.next != None:
                headIndex = i

resultImg = stitchQuad(beforeSort[headIndex].img,
                       beforeSort[headIndex].next.img,
                       beforeSort[headIndex].next.next.img,
                       beforeSort[headIndex].next.next.next.img)

print("영상 순서 : %d - %d - %d - %d"% (beforeSort.index(beforeSort[headIndex])+1, beforeSort.index(beforeSort[headIndex].next)+1, beforeSort.index(beforeSort[headIndex].next.next)+1, beforeSort.index(beforeSort[headIndex].next.next.next)+1))

cv2.imshow("Stitching",resultImg)
                 
cv2.waitKey()
cv2.destroyAllWindows()
                    
