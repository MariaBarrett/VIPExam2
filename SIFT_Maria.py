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

def detectcompute(data):
	descriptors = []
	keypoints =[]
	for img in data: 
		image = cv2.imread(img)
		gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(gray,None)
		keypoints.append(kp)
		descriptors.append(des)
	return keypoints, descriptors

train_kp1, train_des1 = detectcompute(train1)
train_kp2, train_des2 = detectcompute(train2)

traindescriptor = train_des1+train_des2

#getting a np.array of descriptors
traindescriptor1 = []
for img in traindescriptor[::20]:
	for des in img:
		traindescriptor1.append(des)
		traindescriptor2 = np.array(traindescriptor1)

#computing K-Means with K = 2 (2 clusters)
centroids, idx = kmeans2(traindescriptor, 500)
#assign each sample to a cluster
#idx,_ = vq(traindescriptor2,centroids)
print len(centroids)

print idx
	
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
