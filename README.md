# Face-Mask-Detection-With-Feature-Extraction-and-Incremental-Learning-on-VGG16-Model

A simple application that can detect whether a person is wearing a mask or not in real time or in an image. This model was developed using transfer learning by feature extraction and incremental learning on pre trained VGG16 Model.

# NOTE: Due to large size, all files have not been included here.

STEPS ON HOW TO RUN THIS MODEL

1. Download the dataset from here : https://github.com/prajnasb/observations/tree/master/experiements/data.
   
2. Create two directories named "extracted_features" and "dataset". 
   Your repository should be organized like this.
   
   

3. Now run feature_extraction.py to extract the features from your dataset. The extracted features will be stored in the form of CSV's in the repository "extracted_features".

4. Run train.py to create the face detector.

5. Now to detect face-mask ,you can use either "detect_image.py" or "detect_in_video.py" as per your use.

# OUTPUT::
