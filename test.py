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

def codebook_from_file():
	from_db = open("codebook.txt", "r")
	codebook_from_db = pickle.load(from_db)
	from_db.close()
	return codebook_from_db

cb = codebook_from_file
print type(cb)