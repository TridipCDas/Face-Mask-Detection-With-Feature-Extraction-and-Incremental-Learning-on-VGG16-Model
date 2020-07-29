#Import the necessary packages
from keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

#Import the VGG16 and pre-trained classifier model
modelV = VGG16(weights="imagenet", include_top=False)
model=load_model("trial1.h5")

#Import the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classes=['with-mask','without-mask']

def detect_mask(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.1, minNeighbors=8) 
    
    for (x,y,w,h) in face_rects:
        face=face_img[y:y+h,x:x+w]
        face=cv2.resize(face,(224,224))
        face=img_to_array(face)
        custom= np.expand_dims(face, axis=0)
        custom= preprocess_input(custom)
        feat = modelV.predict(custom)
        feat=feat.reshape((1,7*7*512))
        probs=model.predict(feat)
        prediction = np.argmax(probs, axis=1)
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
        label=classes[prediction[0]]
        color = (0, 255, 0) if label == "with-mask" else (0, 0, 255)
        marker=str(label)+" {:.2f}%".format(float(probs[0][prediction[0]])*100)
        cv2.putText(face_img, marker, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        
    return face_img

def detect_video():
    
    cap = cv2.VideoCapture(0) 
  
    while True: 
        
        ret, frame = cap.read(0) 
        frame = detect_mask(frame)
        cv2.imshow('Video Face Mask Detection', frame) 
        c = cv2.waitKey(1) 
        if c == 27: 
            break 
    
    cap.release() 
    cv2.destroyAllWindows()


detect_video()
