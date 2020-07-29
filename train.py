from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
import pickle
import os



DATASET_LOCATION="dataset"
TRAIN_LOCATION="training"
TEST_LOCATION="test"
CSV_PATH="extracted_features"
classes=['with-mask','without-mask']
BATCH_SIZE=32
LE_PATH ="le.cpickle"

def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
	# open the input file for reading
	f = open(inputPath, "r")

	# loop indefinitely
	while True:
		# initialize our batch of data and labels
		data = []
		labels = []

		# keep looping until we reach our batch size
		while len(data) < bs:
			# attempt to read the next row of the CSV file
			row = f.readline()

			# check to see if the row is empty, indicating we have
			# reached the end of the file
			if row == "":
				# reset the file pointer to the beginning of the file
				# and re-read the row
				f.seek(0)
				row = f.readline()

				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break

			# extract the class label and features from the row
			row = row.strip().split(",")
			label = row[0]
			label = to_categorical(label, num_classes=numClasses)
			features = np.array(row[1:], dtype="float")

			# update the data and label lists
			data.append(features)
			labels.append(label)

		# yield the batch to the calling function
		yield (np.array(data), np.array(labels))

# load the label encoder from disk
le = pickle.loads(open(LE_PATH, "rb").read())

# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([CSV_PATH,"{}.csv".format(TRAIN_LOCATION)])
testPath = os.path.sep.join([CSV_PATH,"{}.csv".format(TEST_LOCATION)])

# determine the total number of images in the training
totalTrain = sum([1 for l in open(trainPath)])

# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, BATCH_SIZE,len(classes), mode="train")
testGen = csv_feature_generator(testPath, BATCH_SIZE,len(classes), mode="eval")

# define our simple neural network
model = Sequential()
model.add(Dense(256, input_shape=(7 * 7 * 512,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(len(classes), activation="softmax"))

# compile the model
opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training simple network...")
H = model.fit_generator(trainGen,steps_per_epoch=totalTrain // BATCH_SIZE,epochs=25)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict_generator(testGen,
	steps=(totalTest //BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs,
	target_names=le.classes_))

model.save('trial1.h5')