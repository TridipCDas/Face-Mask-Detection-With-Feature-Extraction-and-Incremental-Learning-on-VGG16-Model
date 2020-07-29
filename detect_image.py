#Import the necessary packages
from keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Import the VGG16 and pre-trained classifier model
modelV = VGG16(weights="imagenet", include_top=False)
model=load_model("trial1.h5")

#Import the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classes=['with-mask','without-mask']

def detect_mask(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.3, minNeighbors=8) 
    
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



custom=np.uint8(load_img('Nadia_Murad.jpg'))
predicted_image=detect_mask(custom)
plt.figure(figsize=(17,17))
plt.imshow(predicted_image)
