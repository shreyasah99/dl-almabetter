 # importing all necessary libraries

import av
import cv2
import numpy as np
import streamlit as st        
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer     #streamlit-webrtc helps to deal with real-time video streams.

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers


my_model = tf.keras.models.load_model('shreyas_scratch_model.h5')    # loading .h5 file of our model and storing it in my_model

class VideoTransformer(VideoTransformerBase): 
    def transform(self, frame):                                                            # transform() method, which transforms each frame coming from the video stream.
        img = frame.to_ndarray(format="bgr24")                                              # coverting captured image into array of pixels
        face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')          #  load cascade classifier
        #eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        class_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']        # the prediction will be number from 0-6 ; to link it to its emotion we created this dictionary.



        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  # converting image into grayscale
        face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)          # ROI (region of interest of detected face, stored as tuple of bottom left
        if face_roi is ():                                                # check if face_roi is empty ie. no face detected
            return img

        for(x,y,w,h) in face_roi:                                          # iterate through faces and draw rectangle over each face
            x = x - 5
            w = w + 10
            y = y + 7
            h = h + 2
            cv2.rectangle(img, (x,y),(x+w,y+h),(125,125,10), 2)           # (x,y)- top left point  ; (x+w,y+h)-bottom right point  ;  (125,125,10)-colour of rectangle ; 2- thickness 
            img_color_crop = img[y:y+h,x:x+w]                             # croping colour image
            final_image = cv2.resize(img_color_crop, (48,48))           # size of colured image is resized to 224,224
            final_image = np.expand_dims(final_image, axis = 0)           # array is expanded by inserting axis at position 0
            final_image = final_image/255.0                               # feature scaling of final image
            prediction = my_model.predict(final_image)                    # emotion of the captured image is detected with the help of our model
            label=class_labels[prediction.argmax()]                       # we find the label of class which has maximaum probalility 
            cv2.putText(img,label, (50,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (120,10,200),3)    
                                                      # putText is used to draw a detected label on image
                                                      # (50,60)-top left coordinate   FONT_HERSHEY_SCRIPT_COMPLEX-font type
                                                      # 2-fontscale   (120,10,200)-font colour   3-font thickness
       
        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)      
# image captured from webcam is sent to VideoTransformer function
#webrtc_streamer can take video_transformer_factory argument, which is a Callable that returns an instance of a class which has transform(self, frame) method.
        
         

