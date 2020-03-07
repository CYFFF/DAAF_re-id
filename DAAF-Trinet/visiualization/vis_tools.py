import skimage
import matplotlib

import skimage.io
import skimage.transform
import numpy as np
import cv2

from matplotlib import pyplot as plt
# from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

import tensorflow as tf


def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))