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


"""detectcompute(data,x_train)
This function takes data and the class label x_train.

"""
def detectcompute(data):
	for img in data: 
		image = cv2.imread(img)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)

		"""
		Not really sure why we need this..
		for d in des:
			np.insert(d,[0],x_train) #I'm trying to insert the class label as the first value of every descriptor. 
			#np.insert(d,slice(0),x_train)
		"""
	return des


train_des1 = detectcompute(train1,1)
train_des2 = detectcompute(train2,2)

X_train = np.concatenate((train_des1, train_des2),axis=0)


#computing K-Means 
codebook,distortion = kmeans(whiten(X_train),5)
idx,distor = vq(X_train,codebook)
print idx
