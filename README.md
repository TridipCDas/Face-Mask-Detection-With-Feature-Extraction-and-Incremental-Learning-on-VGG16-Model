# Face-Mask-Detection-With-Feature-Extraction-and-Incremental-Learning-on-VGG16-Model

A simple application that can detect whether a person is wearing a mask or not in real time or in an image. This model was developed using transfer learning by feature extraction and incremental learning on pre trained VGG16 Model.

#### NOTE: Due to large size, all files have not been included here.

STEPS ON HOW TO RUN THIS MODEL

1. Download the dataset from here : https://github.com/prajnasb/observations/tree/master/experiements/data.
   
2. Create two directories named "extracted_features" and "dataset". 
   Your repository should be organized like this.
   
   ![Screenshot (343)](https://user-images.githubusercontent.com/40006730/88848023-7f432e00-d205-11ea-9212-fd1818adc3dc.png)
   
   

3. Now run feature_extraction.py to extract the features from your dataset. The extracted features will be stored in the form of CSV's in the repository "extracted_features".

4. Run train.py to create the face detector.

5. Now to detect face-mask ,you can use either "detect_image.py" or "detect_in_video.py" as per your use.

##### OUTPUT::

![op1](https://user-images.githubusercontent.com/40006730/88848075-94b85800-d205-11ea-8478-a5f18c1ea883.png)

![op2](https://user-images.githubusercontent.com/40006730/88848092-9b46cf80-d205-11ea-8fc5-8e95362c3852.png)

![op3](https://user-images.githubusercontent.com/40006730/88848104-a0a41a00-d205-11ea-80e9-65f32e4b90b4.png)

