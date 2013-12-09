import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq,whiten
import pylab as pl

# Extracting test and train set
path1 = glob.glob('../VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('../VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')
#RELATIVE PATHS!! ^_^


train1 = path1[:30]
train2 = path2[:30]
train1.extend(train2) #One list only please!

test1 = path1[30:]
test2 = path2[30:]

# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()

#--------------------------------------------------------------------------
#Detection


"""detectcompute(data,x_train)
This function takes data and the class label x_train.
It then calculates the SIFT descriptors for every image and returns all image descriptors as rows in a single array.

"""

def detectcompute(data):
	descr = []

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		descr.append(des)
	
	out = np.vstack(descr) #Vertical stacking of our descriptor list. Genius function right here.
	return out



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

def index(data,label): #Python does not like when you use already defined names such as "class" or "def" to create new variables.
	"Beautiful function goes here?"
