# Import section
import csv
import cv2
import numpy as np
import sklearn

# Import Keras layers and model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

steering_correction = 0.2 # Select steering correction in case of the left and right cameras
batch_size= 128 # Select batch size for generator
number_cameras = 3 # select the left, right and center cameras
left_camera_idx = 1 # column index of the left camera data in the data file
right_camera_idx = 2 # column index of the right camera data in the data file
def driving_log_read(data_path):
	'''
	This function reads the data log and extract the images paths line by line. Also it loads
	the measurements information.
	'''
	print("Reading images")
	image_paths = [] # Array to save the image paths
	measurements = [] # Array to save the angles
	first_line = True
	with open(data_path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			if first_line:
				first_line = False
				continue
			# Gets the paths of the 3 cameras
			for image_index in range(number_cameras):
				source_path = line[image_index]
				filename =  source_path.split('/')[-1]
				current_path = './data/IMG/' + filename
				image_paths.append(current_path)
				measurement = float(line[3])
				# Adjust steering depending on the camera
				if image_index == left_camera_idx:
					measurements.append(measurement + steering_correction)
				elif image_index == right_camera_idx:
					measurements.append(measurement - steering_correction)
				else:
					measurements.append(measurement)
	print("Images loaded")
	return image_paths, measurements


def flip_images(images, measurements):
	'''
	This function flips the images samples one by one so the data set is augmented by 2
	'''
	augmented_images, augmented_measurements = [], []
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image, 1))
		augmented_measurements.append(measurement*-1)
	return augmented_images, augmented_measurements


def generator_define(file_names, measurements, batch_size=128):
	'''
	Defines the generator to create batches of 128 and train them, increasing efficiency because
	the batch is loaded in memory and not the full train set
	'''
	num_samples = len(file_names)
	while 1:
		# Start selecting batch_size-images
		for offset in range(0, num_samples, batch_size):
			file_names_batch = file_names[offset:offset+batch_size]
			measurements_batch = measurements[offset:offset+batch_size]
			images = []
			angles = []
			# Load the image and angle
			for imagePath, measurement in zip(file_names_batch, measurements_batch):
				originalImage = cv2.imread(imagePath)
				image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
				images.append(image)
				angles.append(measurement)
			# Augment the data
			images, angles = flip_images(images, angles)
			# Convert the data to numpy arrays
			X_train_batch = np.array(images)
			y_train_batch = np.array(angles)
			# Yield the generator
			yield sklearn.utils.shuffle(X_train_batch, y_train_batch)



def nvidia_model_define():
	'''
	Defines the training model.
        Taken from http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        paper.
        I decided to use dropouts in the fully connect layers to reduce overfitting.
	'''
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
	model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
	model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	return model

# Read paths and measurements from the driving log
image_paths, measurements = driving_log_read('data/driving_log.csv')

# Split samples, use 80% for train and 20% for validation
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, measurements, test_size=0.2, random_state=42)

# Create generators
train_generator = generator_define(X_train, y_train, batch_size)
validation_generator = generator_define(X_valid, y_valid, batch_size)

# Create nvidia model
model = nvidia_model_define()
# Fit the generator and train the model
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(X_train), validation_data=validation_generator, \
                 nb_val_samples=len(X_valid), nb_epoch=10, verbose=1)
# Save the model
model.save('model.h5')

# View training loss and validation loss 
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
