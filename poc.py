# coding: utf-8
import picamera
import picamera.array
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

import time

camera = picamera.PiCamera()
stream = picamera.array.PiRGBArray(camera)
camera.resolution = (320, 240)
# stream.arrayにRGBの順で映像データを格納
camera.capture(stream, 'bgr', use_video_port=True)
# グレースケールに変換
image = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

stream.seek(0)
stream.truncate()

# stream.arrayにRGBの順で映像データを格納
camera.capture(stream, 'bgr', use_video_port=True)
# グレースケールに変換
image2 = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

stream.seek(0)
stream.truncate()

start = time.time()
shift, error, diffphase = register_translation(image, image2, 100)
print("Exec Time:{:.1f}".format(time.time() - start) + "[sec]")
print("Detected subpixel offset (y, x): {}".format(shift))