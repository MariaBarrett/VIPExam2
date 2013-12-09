import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq, kmeans2
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


"""detectcompute(data,x)
This function takes data and the class label x.

"""
def detectcompute(data,x):
	x = np.array([x],dtype="float32")

	for img in data: 
		image = cv2.imread(img)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)

		"""
		Not really sure why we need this..
		for d in des:
			np.insert(d,[0],x) #I'm trying to insert the class label as the first value of every descriptor. 
			#np.insert(d,slice(0),x)
		"""
	return kp, des


train_kp1, train_des1 = detectcompute(train1,1)
train_kp2, train_des2 = detectcompute(train2,2)

listofdes = np.concatenate((train_des1, train_des2),axis=0)

""" Not really sure if this is what we need, but it is from sklearn

"""
def kmeans(data,n_clusters):
	k_means = cluster.KMeans(n_clusters=n_clusters) #n_init & max_iter is 10
	k_means.fit(data)
	X = k_means.cluster_centers_.squeeze()
	y = k_means.labels_

	return X,y

result = kmeans(listofdes,2)
print result


"""
#computing K-Means 
#centroids, idx = kmeans2(X, 10)
#assign each sample to a cluster
#idx,_ = vq(traindescriptor2,centroids)


# some plotting using numpy's logical indexing
plot(des[idx==0,0],des[idx==0,1],'ob',
     des[idx==1,0],des[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()


Y = pdist(traindescriptor1)
Z = linkage(Y)
dendrogram(Z)

"""
