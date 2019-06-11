# USAGE
# python train.py

# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from pyimagesearch import config
import numpy as np
import pickle
import os

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
le = pickle.loads(open(config.LE_PATH, "rb").read())

# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN)])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST)])

# determine the total number of images in the training and validation
# sets
totalTrain = sum([1 for l in open(trainPath)])

# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="train")
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")

# define our simple neural network
model = Sequential()
model.add(Dense(128, input_shape=(7 * 7 * 512,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(len(config.CLASSES), activation="softmax"))

# compile the model
opt = Adam(r=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training simple network...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	# validation_data=None,
	# validation_steps=None, #totalVal // config.BATCH_SIZE,
	epochs=25)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report

#lets plot the train and val curve
import matplotlib.pyplot as plt

#get the details form the history object
acc = H.history['acc']
loss = H.history['loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.title('Training accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()

plt.show()


print("[INFO] evaluating network...")
predIdxs = model.predict_generator(testGen,
	steps=(totalTest //config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs,
	target_names=le.classes_))

