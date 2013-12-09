import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq,whiten
#from hcluster import pdist, linkage, dendrogram What is this?
import pylab as pl
from sklearn import cluster

# Extracting test and train set
path1 = glob.glob('../VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('../VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')

train1 = path1[:30]
train2 = path2[:30]
test1 = path1[30:]
test2 = path2[30:]

# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()

def detectcompute(data):
	for img in data: 
		image = cv2.imread(img)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
	return des

def detectcompute2(img):
	#Same as detectcompute just doesn't loop through a list of images
	image = cv2.imread(img)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(gray,None)
	return des


train_des1 = detectcompute(train1)
train_des2 = detectcompute(train2)

X_train = np.concatenate((train_des1, train_des2),axis=0)

#computing K-Means 
codebook,distortion = kmeans(whiten(X_train),5)
idx,distor = vq(X_train,codebook)
#print idx

#--------------------------------------------------------------------------
#Indexing

def index(data,x):
#x is the class 
	indexlist = []
	temp = []
	temp2 = []
	for img in data:
		#img.split("/")
		temp.append(x)
		temp.append(img[-14:]) # filename
		des = detectcompute2(img) #calling the function that returns descriptors
		codebook,distortion = kmeans(whiten(des),5)
		idx,distor = vq(des,codebook)
		idx.tolist() #doesn't work. It's just ignored. They are still numpy arrays
		des.tolist()
		for i in range(len(des)):
			temp2.append(idx[i])
			temp2.append(des[i])
		temp.append(temp2) 
	indexlist.append(temp)
	print indexlist[0][0] #print class of first image
	print indexlist[0]


index(train1,2)
	
