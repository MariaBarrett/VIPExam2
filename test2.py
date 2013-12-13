from __future__ import division 

counter = 0
imagepath = "../VIPExam2/101_ObjectCategories/lobster/image_0006.jpg"

resultpath = ['../VIPExam2/101_ObjectCategories/lobster/image_0002.jpg', '../VIPExam2/101_ObjectCategories/lobster/image_0038.jpg', '../VIPExam2/101_ObjectCategories/brontosaurus']

l = imagepath.find("lobster")
b = imagepath.find("brontosaurus")

if l < 0: 
	label = "brontosaurus"
else:
	label = "lobster"

for r in resultpath:
	if r.find(label) > 0:
		counter +=1
	result = counter / len(resultpath)

print ('Result for %s: \n' %imagepath)
print ('Precision rate in top 9: %s' %result)
