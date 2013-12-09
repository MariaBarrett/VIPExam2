from __future__ import division
import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq,whiten
import pylab as pl

# Extracting test and train set
path1 = glob.glob('../VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('../VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')

train1 = path1[:30]
train2 = path2[:30]
train1.extend(train2) #One list only please!

test1 = path1[30:]
test2 = path2[30:]

# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()

#--------------------------------------------------------------------------
#Detection

def detectcompute(data):
	descr = []

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		descr.append(des)
	
	out = np.vstack(descr) #Vertical stacking of our descriptor list. Genius function right here.
	return out


def detectcompute2(img):
	#Same as detectcompute just doesn't loop through a list of images
	image = cv2.imread(img)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(gray,None)
	return des


"""singledetect(data)
This function takes a list of image paths and outputs each images' SIFT descriptors.

"""

def singledetect(data):
	
	for i in range(len(data)):
		des = detectcompute[i]

	return 

### We compute the SIFT descriptors for our entire training set at once and run kmeans on it
X_train = detectcompute(train1)

#computing K-Means 
codebook,distortion = kmeans(whiten(X_train),5)


#### We then compute the SIFT descriptors for every image seperately
#### as to get every images bag of w~erds
idx,distor = vq(X_train,codebook)

#--------------------------------------------------------------------------
#Indexing

def index(data,x):
#x is the class. This function returns a list of lists of lists. The structure is as follows:
#[class, filename, [index of first descriptor, [first descriptor], [index of second decriptor [second descriptor]]]]  
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
		temp2 =[] 
	indexlist.append(temp)
	temp=[]
	print indexlist[0][0] #print class of first image
	print indexlist[0]


index(train1,2)
	

