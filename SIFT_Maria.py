import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans,vq, kmeans2
from hcluster import pdist, linkage, dendrogram
from pylab import plot,show

path1 = glob.glob('/Users/Maria/Documents/ITandcognition/Github/VIPExam2/101_ObjectCategories/lobster/*.jpg')
path2 = glob.glob('/Users/Maria/Documents/ITandcognition/Github/VIPExam2/101_ObjectCategories/brontosaurus/*.jpg')

train1 = path1[:30]
train2 = path2[:30]
test1 = path1[30:]
test2 = path2[30:]

sift = cv2.SIFT()

def detectcompute(data,x):
#this function takes data and the class label, x. 
	x = np.array([x])
	for img in data: 
		image = cv2.imread(img)
		gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		for d in des:
			np.insert(d,[0],x) #I'm trying to insert the class label as the first value of every descriptor. 
			#np.insert(d,slice(0),x)
	return kp, des

train_kp1, train_des1 = detectcompute(train1,1)
train_kp2, train_des2 = detectcompute(train2,2)

print train_kp1[0]

print len(train_des1)
print len(train_des1[0])

listofdes = np.concatenate((train_des1, train_des2),axis=0)

#computing K-Means 
centroids, idx = kmeans2(listofdes, 10)
#assign each sample to a cluster
#idx,_ = vq(traindescriptor2,centroids)
	
"""
# some plotting using numpy's logical indexing
plot(des[idx==0,0],des[idx==0,1],'ob',
     des[idx==1,0],des[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()


Y = pdist(traindescriptor1)
Z = linkage(Y)
dendrogram(Z)

"""
