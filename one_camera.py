import os
import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("/usr/local/cuda-10.0/bin")
sys.path.append("/usr/local/cuda-10.0/lib64")
sys.path.append("/home/cv001/protobuf/bin")
import time
import argparse
import logging
import numpy as np
from numpy.linalg import norm
#import tensorflow as tf
import cv2
#import pika
import  json
#import pycuda.autoinit  # This is needed for initializing CUDA driver
import requests
import zipfile
import subprocess
from datetime import datetime
import multiprocessing
from pypylon import pylon
import six.moves.urllib as urllib
import tarfile
from math import sqrt
import copy
import logging
from pylab import *


def enhance_contrast(image, value):
	image = np.array(image, dtype = np.uint8)
	maxIntensity = 255.0 # depends on dtype of image data
	x = arange(maxIntensity) 

	# Parameters for manipulating image data
	phi = 1
	theta = 1

	# Increase intensity such that
	# dark pixels become much brighter, 
	# bright pixels become slightly bright
	newImage = (maxIntensity/phi)*(image/(maxIntensity/theta))**value
	newImage = array(newImage,dtype=uint8) 

	return newImage

def enhance_saturation(image, saturation = 3, brightness = 1):
	image = np.array(image, dtype = np.uint8)
	hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	#multiple by a factor to change the saturation
	hsvImg[...,1] = hsvImg[...,1]* saturation

	#multiple by a factor of less than 1 to reduce the brightness 
	hsvImg[...,2] = hsvImg[...,2]* brightness

	image=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
	return image


cnt = 0

prev_time = 0
new_time = 0
avg_fps = 1.0
info = pylon.DeviceInfo()
info.SetSerialNumber("22804741")
img = pylon.PylonImage()
#Camera Initialization
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
camera.Open()
pylon.FeaturePersistence.Load("/home/jeff/Pylon features/daA2500-14uc_22804741.pfs", camera.GetNodeMap(), True)

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 640))
start = False
#Start Icount Application
while camera.IsGrabbing():
	grabResult = camera.RetrieveResult(50000, pylon.TimeoutHandling_ThrowException)
	
	if grabResult.GrabSucceeded():
		key = cv2.waitKey(1)
    	#This is for Basler camera
			
		image = converter.Convert(grabResult)
		frame = image.GetArray()
		frame = enhance_saturation(frame, 3, 1)
		frame = enhance_contrast(frame, 0.5)
		# calcuate FPS
		alpha = 0.9
		new_time = time.time()
		fps = 1/(new_time-prev_time)		
		prev_time = new_time
		#exponential moving average
		avg_fps = alpha * avg_fps + (1.0 - alpha) * fps
		avg_fps = int(avg_fps)
		#avg_fps = str(avg_fps)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, str(avg_fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
		cv2.imshow("Left  Right", cv2.resize(frame, (600, 600)))
		
		if key == 27:  # ESC key: quit program
			cv2.destroyAllWindows()
			#out.release()
			break

		elif key == ord('s'):
		#elif cnt%5000 ==0 and start:
			
			cv2.imwrite("pylon/A1_obj_box_a3_%05d.png"%(cnt), frame)
			cnt += 1
		
			
	grabResult.Release()
	


camera.StopGrabbing()
cv2.destroyAllWindows()
			