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
import pickle

# Extracting test and train set

print "=" * 60
print "Initializing the script.\n"

path1 = glob.glob('../VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('../VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')

train1 = path1[:30]
train2 = path2[:30]
train1.extend(train2) #One list only please!

test1 = path1
test2 = path2
test1.extend(test2)
# Defining classifiers as variables and other useful variables
sift = cv2.SIFT()
k = 5
#--------------------------------------------------------------------------
#Detection



"""detectcompute(data)
This function takes a list of image paths as input.
It then calculates the SIFT descriptors for the entire data input and returns all descriptors as rows in a single array.
"""
def detectcompute(data):
	descr = []

	print "="*60,"\n"
	print "Locating all SIFT descriptors for the train set."

	for i in range(len(data)): 
		image = cv2.imread(data[i])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		descr.append(des)
	
	out = np.vstack(descr) #Vertical stacking of our descriptor list. Genius function right here.
	print "Done.\n"
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

	print "Done.\n"
	return sd



"""createdatabase()
This function takes no direct input, but utilizes the image paths assigned at the beginning of the script.
It then computes the K-means clustering on the specified training set and calculates vector quantization on every images descriptor up against the clusters.
It then calls the Bag-of-Visual-Words function. Finally it outputs every image's path and normalized histogram into one file, and the codebook into another file.
"""
def createdatabase():
	X_train = detectcompute(train1)

	print "Clustering the data with K-means"
	codebook,distortion = kmeans(whiten(X_train),k)
	print "Done.\n"
	
	imtrain = singledetect(test1)
	Pdatabase = bow(imtrain,codebook,k) #Pseudo database with list structure


	#Writing to html.table
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
	print "Done.\n"

	codebook_to_file(codebook)


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

	print "Done.\n"
	return out


"""
codebook_to_file()
This function saves the codebook to a file.
"""
def codebook_to_file(codebook):
	print "Saving codebook to file"
	codebookfile = open("codebook.txt", "r+")
	pickle.dump(codebook, codebookfile)
	
	codebookfile.close()
	print "Done.\n"



"""
codebook_from_file()
This function retrieves the codebook from the file.
It returns the codebook,
"""

def codebook_from_file():
	from_db = open("codebook.txt", "r")
	codebook_from_db = pickle.load(from_db)

	from_db.close()
	return codebook_from_db



"""
from_database(path_to_db, filename)
This function retrieves and returns everything from our database: filenames and the adjacent histogram.
These are structured in a nested list like this: [[filename, hisogram],[filename,histogram]...]
"""
def from_database():
	database =[]
	htmldoc = open("table.htm","r") 
	db = BeautifulSoup(htmldoc)
	table = db.find('table')
	for i in range(60):
		temp = []
		filename = table.find('td')
		temp.append(filename.text)
		hist = filename.findNext('td') 
		temp.append(ast.literal_eval(hist.text[7:]))
		database.append(temp)

	htmldoc.close()
	return database



"""Bhattacharyya(one query image, a database)
This function takes a single image and an image database as its input.
It then tries to match the image with every image in the database by measuring the Bhattacharyyan distance between them.
It then returns the 9 closests matches.
"""
def Bhattacharyya(queryimage,db):
    count=[]
    amount=0

    print "Calculating the Bhattacharyya distance."

    for num in range(len(db)):
        for i in range(k):
           amount+=sqrt(queryimage[2][i]*db[num][1][i]) 
        count.append(amount)
        amount=0
        
    for key in range(len(count)-1):
        for x in range(len(count)-key-1):
            if count[x]>count[x+1]:
                count[x],count[x+1]=count[x+1],count[x]
                db[x],db[x+1]=db[x+1],db[x]
                
    queryresult=[]
    for j in range(9):
        queryresult.append(db[j][0])

    print "Done.\n"
    return queryresult



"""userinput()
This function is called at the beginning and takes a user input.
The input is then used as an ouput to call the commands(cmd) function.
"""
def userinput():
	print "="*60
	print "Please select one of the following 3 options. \n"
	print "-"*45
	print "1. Compute the database."
	print "2. Retrieve an image."
	print "3. Exit"
	print "-"*45
	usercmd = raw_input("Choose an option: ")
	commands(usercmd)



"""commands(cmd)
This function takes an input of 

"""
def commands(cmd):
	legal = ["1","2","3"]

	if cmd not in legal:
		print "Invalid input. Please use the numerical value.\n"
		userinput()

	elif cmd == "1":
		createdatabase()
		print "A database has been created. \n" 
		userinput()

	elif cmd == "2":
		print "A retrieval has been made."
		userinput()

	elif cmd == "3":
		print "Quit succesfully."
		raise SystemExit()



"""main()
Starts the programme by calling the userinput-function
"""
def main():
    print ">>> Content Based Image Retrieval by Maria, Guangliang and Alexander \n Vision and Image Processing assignment 2 \n";
    userinput();




#----------------------------------------------------------------------------

#Retrieval
#database = from_database()

#This should be at the bottom, but since we still need to convert retrieval into functions this will serve as the bottom.
if __name__ =='__main__':
    main(); 
"""
print "="*60
print "Retrieving matches in our database for a random image"

query = singledetect(test1)
querybow = bow(query, codebook_from_db,k)


# create the bag of visual words for the query image
# query image is created randomly
queryimage=querybow[random.randint(0,len(test1)-1)]

resultpath = Bhattacharyya(queryimage, database)

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
"""