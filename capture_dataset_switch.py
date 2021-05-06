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
import threading
import argparse
from pylab import *
from numba import jit


info = pylon.DeviceInfo()

parser = argparse.ArgumentParser(description='saturation')
parser.add_argument('--saturation', type=float,
                    help='saturation')

args = parser.parse_args()
saturation = args.saturation

def camera_set(serial_number, feature):

	info.SetSerialNumber(serial_number)
	img = pylon.PylonImage()
	#Camera Initialization
	camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
	camera.Open()
	pylon.FeaturePersistence.Load(feature, camera.GetNodeMap(), True)

	# Grabing Continusely (video) with minimal delay
	camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

	return camera

def catch_camera(camera):
	result = camera.RetrieveResult(50000, pylon.TimeoutHandling_ThrowException)
	return result

def controller(grabResult, open, close, key, k):
	if key == ord(open):
		k = 1
		print("camera start")

	elif key == ord(close):
		k = 0
		print("camera stop")

	if k == 1:
		image = converter.Convert(grabResult)
		frame = image.GetArray()
		frame = cv2.resize(frame, (600, 600))
	elif k == 0 :
		frame = np.zeros(shape = (600, 600, 3))	

	return frame, k



def enhance_saturation(image, saturation = 3, brightness = 1):
	image = np.array(image, dtype = np.uint8)
	hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	#multiple by a factor to change the saturation
	hsvImg[...,1] = hsvImg[...,1]* saturation

	#multiple by a factor of less than 1 to reduce the brightness 
	hsvImg[...,2] = hsvImg[...,2]* brightness

	image=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
	return image



def remove_shadow(image, dilated, blur):
	image = np.array(image, dtype = np.uint8)
	rgb_planes = cv2.split(image)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
		dilated_img = cv2.dilate(plane, np.ones((dilated,dilated), np.uint8))
		bg_img = cv2.medianBlur(dilated_img, blur)
		diff_img = 255 - cv2.absdiff(plane, bg_img)
		norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		result_planes.append(diff_img)
		result_norm_planes.append(norm_img)

	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)
	return result


def lower_explosure(image):
	image = np.array(image, dtype = np.uint8)
	chans = cv2.split(image)
	colors = ("b", "g", "r")

	for (chan, color) in zip(chans, colors):
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

	equalizeOver = np.zeros(image.shape, image.dtype)
	equalizeOver[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
	equalizeOver[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
	equalizeOver[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

	return equalizeOver

def exponential_moving_average(fps, avg_fps):
	
	alpha = 0.9
	#exponential moving average
	avg_fps = alpha * avg_fps + (1.0 - alpha) * fps
	avg_fps = int(avg_fps)
	#avg_fps = str(avg_fps)

	return avg_fps

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

def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

camera = camera_set("22804747", "/home/jeff/Pylon features/daA2500-14uc_22804741.pfs")
camera2 = camera_set("22804741", "/home/jeff/Pylon features/daA2500-14uc_22804741.pfs")

converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 640))
start = False
#Start Icount Application


def capture(camera, camera2):
	#initial setup
	prev_time = 0
	new_time = 0
	avg_fps = 20.0
	avg_fps_1 = camera.ResultingFrameRate.GetValue()
	avg_fps_2 = camera2.ResultingFrameRate.GetValue()
	cnt1 = 0
	cnt2 = 0
	k1 = 0
	k2 = 0
	while True:
		#multiprocessing
		#grab image
		grabResult = catch_camera(camera)
		grabResult2 = catch_camera(camera2)
		fps_1 = camera.ResultingFrameRate.GetValue()
		fps_2 = camera2.ResultingFrameRate.GetValue()

		if grabResult.GrabSucceeded() & grabResult2.GrabSucceeded():
			#This is for Basler camera
			key = cv2.waitKey(1)
			frame1, k1 = controller(grabResult, 'f', 'c', key, k1)
			frame2, k2= controller(grabResult2, 's', 'd', key, k2)


			frame1 = enhance_saturation(frame1, saturation)
			#frame1 = cv2.fastNlMeansDenoisingColored(frame1,None,3,3,7,21)
			#frame2 = enhance_saturation(frame2, saturation)
			
			b = brightness(frame1)
			print(b)
			if b < 100:
				value  = 0.3
			elif b < 150 and b > 100:
				value = 0.5
			elif b > 150:
				value = 1
			
			frame1 = enhance_contrast(frame1, value)
			#frame2 = enhance_contrast(frame2, value)
			# frame1 = lower_explosure(frame1)
			
			

			#This is for Basler camera
			frame1 = np.uint8(frame1)
			frame2 = np.uint8(frame2)
			
			# calcuate FPS
			avg_fps_1 = exponential_moving_average(fps_1, avg_fps_1)
			avg_fps_2 = exponential_moving_average(fps_2, avg_fps_2)
			
			new_time = time.time()
			fps = 1.0 /(new_time - prev_time)
			prev_time = new_time
			
			#print(fps)
			
			
			avg_fps = exponential_moving_average(fps, avg_fps)
			frames = np.concatenate((frame2, frame1), axis=1)
			font = cv2.FONT_HERSHEY_SIMPLEX

			cv2.putText(frames, str(avg_fps_1), (len(frames)+ 7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
			cv2.putText(frames, str(avg_fps_2), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
			cv2.putText(frames, str(avg_fps), (7, 550), font, 3, (0, 100, 255), 3, cv2.LINE_AA)
			cv2.imshow("Left  Right", cv2.resize(frames, (1200, 600)))
			#cv2.imshow(name, cv2.resize(frame, (600, 600)))
			
			if key == 27:  # ESC key: quit program
				cv2.destroyAllWindows()
				#out.release()
				break
			
			elif key == ord('r'):
			#elif cnt%5000 ==0 and start:
				
				cv2.imwrite('pylon/camera1_obj_box_a3_%05d.png'%(cnt1), frame1)
				cnt1 +=1

				cv2.imwrite('pylon/camera2_obj_box_a3_%05d.png'%(cnt2), frame2)
				cnt2 +=1
			

		grabResult.Release()



capture(camera, camera2)	
	
camera.StopGrabbing()
camera2.StopGrabbing()
cv2.destroyAllWindows()
				
