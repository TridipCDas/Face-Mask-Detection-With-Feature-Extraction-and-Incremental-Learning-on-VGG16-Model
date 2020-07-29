from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
from imutils import paths
import random
import numpy as np
import pickle


DATASET_LOCATION="dataset"
TRAIN_LOCATION="training"
TEST_LOCATION="test"
CSV_PATH="extracted_features"
classes=['with-mask','without-mask']
BATCH_SIZE=32
LE_PATH ="le.cpickle"

# load the VGG16 network and initialize the label encoder
model = VGG16(weights="imagenet", include_top=False)
le = None


#Function for preprocessing the image before feeding into the model
def preprocess_image(imagePath):
    image=load_img(imagePath,target_size=(224,224))
    
    image = img_to_array(image)
    
    image = np.expand_dims(image, axis=0)
	 
    image = preprocess_input(image)
    
    return image

  
for split in (TRAIN_LOCATION,TEST_LOCATION):
    imagePaths=[]
    labels=[]
    
    for p in classes:
        path=os.path.sep.join([DATASET_LOCATION,split,p])
        iPath=list(paths.list_images(path))
        imagePaths=imagePaths+iPath
        labels=labels+[p for i in range(len(iPath))]
    
    
    #Random shuffling the imagePaths and the labels
    temp = list(zip(imagePaths, labels)) 
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    imagePaths=list(res1)
    labels=list(res2)
    
    # if the label encoder is None, create it
    if le is None:
        le=LabelEncoder()
        le.fit(labels)
        
    #open the csv file for writing
    csvPath = os.path.sep.join([CSV_PATH,"{}.csv".format(split)])
    csv = open(csvPath, "w")
     
    
    
     # loop over the images in batches  
    for (b, i) in enumerate(range(0, len(imagePaths),BATCH_SIZE)):
         print("[INFO] processing batch {}/{}".format(b + 1,int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))))
         batchPaths = imagePaths[i:i + BATCH_SIZE]
         batchLabels = le.transform(labels[i:i + BATCH_SIZE])
         batchImages = []
         
         # loop over the images and labels in the current batch
         for imagePath in batchPaths:
             
             image =preprocess_image(imagePath) 
             # add the image to the batch
             batchImages.append(image)
        # pass the images through the network and use the outputs as
        # our actual features, then reshape the features into a
        # flattened volume
         
         batchImages = np.vstack(batchImages)
         features = model.predict(batchImages, batch_size=BATCH_SIZE)
         features = features.reshape((features.shape[0], 7 * 7 * 512))
        
                
         
         # loop over the class labels and extracted features 
         for (label, vec) in zip(batchLabels, features):
            # construct a row that exists of the class label and  extracted features
            vec = ",".join([str(v) for v in vec])
            csv.write("{},{}\n".format(label, vec))    
   
    #close the CSV file
    csv.close()
          
# serialize the label encoder to disk
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()        
           
         
         
         
            
    
    
        
    
    
    
        
        
        
        