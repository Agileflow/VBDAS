# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 04:07:47 2017

@author: abel abiodun
"""

import numpy as np
import cv2
import vrep
import sys


class VBDAS(object):

    def __init__(self,vision_source,clientId,cascade,images):
        self.err, self.visionSensorHandle = vrep.simxGetObjectHandle(clientID, vision_source, vrep.simx_opmode_oneshot_wait)
        self.client = clientId
        self.cascade = cascade

        self.signs = list() # stores signs as numpy array
        self.kpds = list()  # Keypoints and descriptors
        
        for image in images:
            self.signs.append(cv2.imread('signs/'+str(image),0))
            
        
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher()
        
        for sign in self.signs:
            # find the keypoints and descriptors with SIFT
            kp, ds = self.sift.detectAndCompute(sign,None)
            self.kpds.append(ds)
        
        self.process()
        
    def match(self,source):
    
        # find the keypoints and descriptors with SIFT
        kp, des = self.sift.detectAndCompute(source,None)
        
        matches = list()
        
        # BFMatcher with default params
        for ds in self.kpds:
            matches.append(self.bf.knnMatch(des,ds,k=2))
        
        # Apply ratio test
        goodPoints = list()
        i = 0
        
        for match in matches:
            goodPoints.append([])
            for m,n in match:
                if m.distance < 0.75*n.distance:
                    goodPoints[i].append([m])
            i += 1
        i = None
        k = 0
        for good in goodPoints:
            if len(good) > k:
                k = len(good)
                i = good
                
        try:
            i = goodPoints.index(i)
            cv2.imshow('Found', self.signs[i])
        except ValueError as v:
            k = v
    
    def process(self):
        
        self.err, resolution, image = vrep.simxGetVisionSensorImage(self.client, self.visionSensorHandle,0,vrep.simx_opmode_streaming)
         
        # stream video  frames one by one
        try:
            self.err, resolution, image = vrep.simxGetVisionSensorImage(self.client, self.visionSensorHandle,0,vrep.simx_opmode_buffer)
            
            sign_cascade = cv2.CascadeClassifier(self.cascade)
            
            while True:
                self.err, resolution, image = vrep.simxGetVisionSensorImage(self.client, self.visionSensorHandle,0,vrep.simx_opmode_streaming)
                
                if len(resolution) > 0:
                    
                    image = np.array(image, dtype=np.uint8).reshape(resolution[0], resolution[1], 3)
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    sign = sign_cascade.detectMultiScale(gray,10,10)

                    for (x,y,w,h) in sign:
                      #  print('x:', x, '\ty:', y,'\tw:', w, '\th:', h)
                        roi = image[y:y+h+35, x:x+w+35] # extract the object detected
                        self.match(roi)
                        cv2.imshow('Detected', roi)
                        sign = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
                    
                    cv2.imshow('Vision Source', image)
                    
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break
        finally:
            cv2.destroyAllWindows()
            

if __name__ == '__main__': 
    vrep.simxFinish(-1)     # closes any existing open connection
    clientID = vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
        
    if clientID!=-1:
        print ('Connected to remote API server')
        signs = ['5_limit.jpg','light_ahead.jpg', 'pedestrian.jpg','10_limit.png','go.png','stop.png','turn_ahead.jpg']
        VBDAS(vision_source='Vision_front', clientId=clientID, cascade='cascade_xml/signs-cascade.xml', images=signs)
    else:
        print('Connection failed')
        sys.exit('Remote API dropped')
