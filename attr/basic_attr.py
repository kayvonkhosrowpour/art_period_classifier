"""
Filename: basic_attr.py
Author(s): Kayvon Khosrowpour
Date Created: 10/25/18

Description:
Extracts basic attributes from images, like median pixel values and
different color spaces.
"""

import cv2
import numpy as np

def hue_value_stats(bgr_img):
    """
    Given an opencv bgr image, returns the hue_median and value_median
    of the HSV representation of the image.

    Args:
      bgr_img: the opencv bgr image as read from cv2.imread()

    Returns:
      hue_med: the median of the hue channel of the HSV color space
      sat_med: the median of the saturation channel of the HSV color space
      value_med: the median of the value channel of the HSV color space
    """

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV) # convert to hsv
    hue_med = np.median(hsv_img[:,:,2]) # median of hue channel
    sat_med = np.median(hsv_img[:,:,1]) # median of saturation channel
    value_med = np.median(hsv_img[:,:,0]) # median of value (i.e. brightness) channel

    return hue_med, sat_med, value_med
