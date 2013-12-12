from __future__ import division
import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq,whiten
import pylab as pl
from collections import Counter

# Extracting test and train set
print "=" * 60
print "Initializing the script"
print "-"*60
print "Loading images."
path1 = glob.glob('../VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('../VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')
#RELATIVE PATHS!! ^_^

train1 = path1[:30]
train2 = path2[:30]
train1.extend(train2) #One list only please!

test1 = path1[30:]
test2 = path2[30:]
print "Done."

# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()
k = 10
#--------------------------------------------------------------------------
#Detection


"""detectcompute(data)
This function takes a list of image paths as input.
It then calculates the SIFT descriptors for the entire data input and returns all descriptors as rows in a single array.

"""

def detectcompute(data):
	descr = []

	print "Locating all SIFT descriptors for the train set."

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		descr.append(des)
	
	out = np.vstack(descr) #Vertical stacking of our descriptor list. Genius function right here.
	print "Done."
	return out


"""singledetect(data)
This function takes a list of image paths as inputs.
It then outputs each images' path and corresponding SIFT descriptors.

"""

def singledetect(data):
	sd = []
	print "Locating and assigning SIFT descriptors for each image"

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		sd.append([data[i],des])

	print "Done."
	return sd



"""bow(list of images,codebook,clusters)
This function taskes a list of image paths, a codebook and an integer denoting the amount of clusters as input.
It then computes each image's bag of words as a normalized histogram in a pseudo-dictionary.
It then outputs 
"""

def bow(images,codebook,clusters):
	out = images
	temp = []

	print "-"*60
	print "Creating the pseudo database."
	for im in images:
		c = Counter()
		bag,dist = vq(whiten(im[1]),codebook)
		
		for word in bag:
			c[word]+=1

		#Creating histograms
		for i in range(clusters):
			if i not in c.iterkeys():
				c[i] = 0
		temp.append(c)
		
	for i in range(len(temp)):
		out[i].append(temp[i])

	print "Done."
	return out



### We compute the SIFT descriptors for our entire training set at once and run kmeans on it

X_train = detectcompute(train1)

print "-"*60
print "Clustering the data with K-means"
#computing K-Means 
codebook,distortion = kmeans(whiten(X_train),k)


#### We then compute the SIFT descriptors for every image seperately as to get every images bag of words
imtrain = singledetect(train1)

#Pseudo database with list structure
Pdatabase = bow(imtrain,codebook,k)

#--------------------------------------------------------------------------
#Print in table

print "Converting the database into a HTML file"

htmltable = open("table.html","r+") 

begin = "<html><body><table cellpadding=5><tr><th>Filename</th><th>Histogram</th></tr>"
htmltable.write(begin)

for i in range(len(Pdatabase)):
    middle = "<tr><td>%(filename)s</td><td>%(histogram)s</td></tr>" % {"filename": Pdatabase[i][0], "histogram": Pdatabase[i][-1]}
    htmltable.write(middle)

end = "</table></body></html>"    	
htmltable.write(end)

print "Done" 
htmltable.close()
