from __future__ import division
from math import *
from bs4 import BeautifulSoup
from collections import Counter
from scipy.cluster.vq import kmeans,vq,whiten
from PIL import Image

import cv2
import numpy as np
import glob
import pylab as pl
import ast
import random



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

test1 = path1
test2 = path2
test1.extend(test2)
print "Done."

# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()
k = 500
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
			if i in c.iterkeys():
				c[i] = c[i]/sum(c.values())
			if i not in c.iterkeys():
				c[i] = 0
		
		temp.append(c)
		
	for i in range(len(temp)):
		out[i].append(temp[i])

	print "Done."
	return out



"""Bhattacharyya(one query image, a database)
This function takes a single image and an image database as its input.
It then tries to match the image with every image in the database by measuring the Bhattacharyyan distance between them.
It then returns the 9 closests matches.

"""
def Bhattacharyya(queryimage,db):
    count=[]
    amount=0

    print "-"*60
    print "Calculating the Bhattacharyya distance."

    for num in range(len(db)):
        for i in range(k):
           amount+=sqrt(queryimage[2][i]*db[num][2][i]) 
        count.append(amount)
        amount=0
        
    for key in range(len(count)-1): #
        for x in range(len(count)-key-1):
            if count[x]>count[x+1]:
                count[x],count[x+1]=count[x+1],count[x]
                db[x],db[x+1]=db[x+1],db[x] #is this some fancy sorting again? Yes it is, and why are we using it again? Bubble sort performs worse than .sorted()
                
    queryresult=[]
    for j in range(9):
        queryresult.append(db[j][0])

    print "Done."
    return queryresult


### We compute the SIFT descriptors for our entire training set at once and run kmeans on it

X_train = detectcompute(train1)

print "-"*60
print "Clustering the data with K-means"
#computing K-Means 
codebook,distortion = kmeans(whiten(X_train),k)


#### We then compute the SIFT descriptors for every image seperately as to get every images bag of words
imtrain = singledetect(test1)

#Pseudo database with list structure
Pdatabase = bow(imtrain,codebook,k)

#--------------------------------------------------------------------------
#Print in table

print "Converting the database into a HTML file"

htmltable = open("table.htm","r+") 

begin = "<htm><body><table cellpadding=5><tr><th>Filename</th><th>Histogram</th></tr>"
htmltable.write(begin)

for i in range(len(Pdatabase)):
    middle = "<tr><td>%(filename)s</td><td>%(histogram)s</td></tr>" % {"filename": Pdatabase[i][0], "histogram": Pdatabase[i][-1]}
    htmltable.write(middle)

end = "</table></body></html>"    
htmltable.write(end)

htmltable.close()

print "Done." 

#------------------------------------------------------------
#Retrieving from database
"""
htmldoc = open("table.htm","r") 
database = BeautifulSoup(htmldoc)
table = database.find('table')
filename = table.find('td', text='VIPExam2/101_ObjectCategories/lobster/image_0004.jpg') #here I guess you want to insert a variable containing the filename
td = filename.findNext('td') #next td contains the histogram for the specified image
histogram_from_db = ast.literal_eval(td.text[7:]) # using ast to turn the unicode string back into a dictionary
"""
#----------------------------------------------------------------------------

#Retrieval

print "="*60
print "Retrieving matches in our database for a random image"

query = singledetect(test1)
querybow = bow(query, codebook,k)

# create the bag of visual words for the query image
# query image is created randomly
queryimage=querybow[random.randint(0,len(test1)-1)]
resultpath = Bhattacharyya(queryimage, Pdatabase)

print "Done."
print "-"*60
print "Plotting the results."

imageplot=[]#used to store the matched images

for result in resultpath:
    img=np.array(Image.open(result))
    imageplot.append(img)

#plot the query image
pl.imshow(np.array(Image.open(queryimage[0])))

#plot the matched images
for i in range(9):
    pl.subplot(331+i)
    pl.imshow(imageplot[i])
    pl.axis('off')

pl.show()
pl.close()