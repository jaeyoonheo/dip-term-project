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

beforeSort = [img1, img2, img3, img4]
afterSort = [img1, img2, img3, img4]

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
    img1 = cv2.rotate(img1,cv2.ROTATE_90_COUNTERCLOCKWISE)
    img2 = cv2.rotate(img2,cv2.ROTATE_90_COUNTERCLOCKWISE)
    img3 = cv2.rotate(img3,cv2.ROTATE_90_COUNTERCLOCKWISE)
    img4 = cv2.rotate(img4,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    leftImg = stitchImgs(img1, img3)
    rightImg = stitchImgs(img2, img4)
    
    leftImg = cv2.rotate(leftImg,cv2.ROTATE_90_CLOCKWISE)
    rightImg = cv2.rotate(rightImg,cv2.ROTATE_90_CLOCKWISE)
    
    return stitchImgs(leftImg,rightImg)

def compareImg(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
        
    flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    matches = flann.knnMatch(des1,des2,k=2)
        
    x1=0
    x2=0
    y1=0
    y2=0
    cnt=0
        
    good = []
    for m,n in matches :
        if m.distance < 0.90*n.distance:
            good.append(m)
            x1 += kp1[m.queryIdx].pt[0]
            x2 += kp2[m.trainIdx].pt[0]
            y1 += kp1[m.queryIdx].pt[1]
            y2 += kp2[m.trainIdx].pt[1]
            cnt += 1
    
    if len(good)<MIN_MATCH_COUNT:
        return 0

    h,w,channel = img1.shape    
    x1 = x1/cnt/w*100
    y1 = y1/cnt/h*100
    h,w,channel = img2.shape
    x2 = x2/cnt/w*100
    y2 = y2/cnt/h*100

    #array = [[x1,y1],[x2,y2]]
    array = [x2,y2]
    return array

array1 = compareImg(img1, img2)
array2 = compareImg(img1, img3)
array4 = compareImg(img1, img4)
array5 = compareImg(img3, img1)
array3 = [array5, array1, array2, array4]

stval_x = (array3[0][0]+array3[1][0]+array3[2][0]+array3[3][0])/4
stval_y = (array3[0][1]+array3[1][1]+array3[2][1]+array3[3][1])/4
stArray = [stval_x, stval_y]

order = [0,0,0,0]

for i in range(0,4):
    if array3[i][0]< stval_x:
        if array3[i][1] < stval_y:
            afterSort[3] = beforeSort[i]
            order[3]=i+1
        else:
            afterSort[0] = beforeSort[i]
            order[0]=i+1
    if array3[i][0]> stval_x:
        if array3[i][1] < stval_y:
            afterSort[2] = beforeSort[i]
            order[2]=i+1
        else:
            afterSort[1] = beforeSort[i]
            order[1]=i+1

cv2.imshow("1q", afterSort[0])
cv2.imshow("2q", afterSort[1])
cv2.imshow("3q", afterSort[2])
cv2.imshow("4q", afterSort[3])

resultImg = stitchQuad(afterSort[1], afterSort[0], afterSort[2], afterSort[3])

print("영상 순서 : %d - %d - %d - %d"% (order[0], order[1], order[2], order[3]))

cv2.imshow("Stitching",resultImg)
                 
cv2.waitKey()
cv2.destroyAllWindows()
                    
