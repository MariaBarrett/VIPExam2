from __future__ import division
import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq,whiten
import pylab as pl
import random
from math import *
from PIL import Image
from collections import Counter

# Extracting test and train set
path1 = glob.glob('101_ObjectCategories\lobster\*.jpg')
path2 = glob.glob('101_ObjectCategories\brontosaurus\*.jpg')
#RELATIVE PATHS!! ^_^

train1 = path1[:30]
train2 = path2[:30]
train1.extend(train2) #One list only please!

test1 = path1[30:]
test2 = path2[30:]
test1.extend(test2)
# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()
clu = 5
#--------------------------------------------------------------------------
#Detection


"""detectcompute(data)
This function takes a list of image paths as input.
It then calculates the SIFT descriptors for the entire data input and returns all descriptors as rows in a single array.

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
This function takes a list of image paths as inputs.
It then outputs each images' path and corresponding SIFT descriptors.

"""

def singledetect(data):
	sd = []

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		sd.append([data[i],des])

	return sd



"""bow(list of images,codebook,clusters)
This function taskes a list of image paths, a codebook and an integer denoting the amount of clusters as input.
It then computes each image's bag of words as a normalized histogram in a pseudo-dictionary.
It then outputs 
"""

def bow(images,codebook,clusters):
	out = images
	temp = []

	for im in images:
		c = Counter()
		bag,dist = vq(whiten(im[1]),codebook)#vq function Assign codes from a code book to observations
		#whiten function Normalize a group of observations on a per feature basis
		for word in bag:
			c[word]+=1

		#Creating normalized histogram
		for i in range(clusters):
			if i in c.iterkeys():
				c[i] = c[i]/len(codebook)
			if i not in c.iterkeys():
				c[i] = 0
		temp.append(c)
		
	for i in range(len(temp)):
		out[i].append(temp[i])
        
	#print out #For you Maria, so you can see the structure!
	return out



### We compute the SIFT descriptors for our entire training set at once and run kmeans on it
X_train = detectcompute(train1)

#computing K-Means 
codebook,distortion = kmeans(whiten(X_train),clu)

#### We then compute the SIFT descriptors for every image seperately as to get every images bag of words
imtrain = singledetect(train1) #[image1[path,descriptors array].image2[path,descriptors array] etc.]

#Pseudo database with list structure
Pdatabase = bow(imtrain,codebook,clu)
#idx,distor = vq(X_train,codebook)

#--------------------------------------------------------------------------
#Print in table

htmltable = open("table.htm","r+") 

begin = "<htm><body><table cellpadding=5><tr><th>Filename</th><th>Histogram</th></tr>"
htmltable.write(begin)

for i in range(len(Pdatabase)):
    middle = "<tr><td>%(filename)s</td><td>%(histogram)s</td></tr>" % {"filename": Pdatabase[i][0], "histogram": Pdatabase[i][-1]}
    htmltable.write(middle)

end = "</table></body></html>"    
htmltable.write(end)

htmltable.close() 
#----------------------------------------------------------------------------
#Retrieval

query = singledetect(test1)
querybow = bow(query, codebook,clu)

# create the bag of visual words for the query image
# query image is created randomly
queryimage=querybow[random.randint(0,len(test1)-1)]
print queryimage[0]
print '+++++++++++++++++++++++='
# caculate the Bhattacharyya disctance between query image and the images in database and output the first 30 matched images
def Bhattacharyya(queryimage,Pdatabase):
    count=[]

    amount=0
    for num in range(len(Pdatabase)):
        for i in range(clu):
           amount+=sqrt(queryimage[2][i]*Pdatabase[num][2][i])
        count.append(amount)
        amount=0
        
    for key in range(len(count)-1):
        for x in range(len(count)-key-1):
            if count[x]>count[x+1]:
                count[x],count[x+1]=count[x+1],count[x]
                Pdatabase[x],Pdatabase[x+1]=Pdatabase[x+1],Pdatabase[x]
                
    queryresult=[]
    for j in range(30):
        queryresult.append(Pdatabase[j][0])#input the path to queryrsult

    return queryresult


resultpath = Bhattacharyya(queryimage, Pdatabase)

imageplot=[]#used to store the matched images

for result in resultpath:
    img=np.array(Image.open(result))
    imageplot.append(img)

#plot the query image
pl.imshow(np.array(Image.open(queryimage[0])))

#plot the matched images
for i in range(30):
    pl.subplot(651+i)
    pl.imshow(imageplot[i])
    pl.axis('off')

pl.show()


