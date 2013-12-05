import cv2
import numpy as np
import glob
from scipy.cluster import vq

path1 = glob.glob('/Users/Maria/Documents/ITandcognition/Github/VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('/Users/Maria/Documents/ITandcognition/Github/VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')

train1 = path1[:30]
train2 = path2[:30]
test1 = path1[30:]
test2 = path2[30:]

sift = cv2.SIFT()

def detectcompute(data):
	descriptors = []
	keypoints =[]
	for img in data: 
		image = cv2.imread(img) # in this line the data is lost...
		gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		keypoints.append(kp)
		descriptors.append(des)
	return keypoints, descriptors

train_kp1, train_des1 = detectcompute(train1)
train_kp2, train_des2 = detectcompute(train2)

traindescriptor = train_des1+train_des2

res, idx = vq.kmeans2(traindescriptor,3)


"""
img1 = cv2.imread('/Users/Maria/Documents/ITandcognition/Github/VIPExam2/101_ObjectCategories/lobster/image_0001.jpg')
gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(gray1,None)

img2 = cv2.imread('/Users/Maria/Documents/ITandcognition/Github/VIPExam2/101_ObjectCategories/brontosaurus/image_0001.jpg')
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(gray2,None)
"""
