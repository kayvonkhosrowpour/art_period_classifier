"""
File: kmeans.py
Author(s): Kayvon Khosrowpour
Date created: 11/3/18

Description:
Calculates entropy of an image.
"""

import numpy as np
import cv2

def get_entropy(bgr_img):
	rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	vals, bins = np.histogram(rgb.reshape(1,-1), 256, [0, 256])
	normvals = vals / np.sum(vals)

	E = 0
	for k in range(0, 256):
		if normvals[k] != 0:
			E += normvals[k] * np.log2(normvals[k])
		# else:
		# 	E -= 100
	E *= -1

	return E
