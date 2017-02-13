import os

import numpy as np
import pandas as pd
import extract
import visualize
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Flatten, Input, Reshape, merge
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K


def merged_model(num_dim, text_dim):
	if K.image_dim_ordering() == 'th':
		text_input_shape = (1, text_dim)	
		num_input_shape = (1, num_dim)
	else:
		text_input_shape = (text_dim, 1)
		num_input_shape = (num_dim, 1)


	# Define the image input
	text_data = Input(shape=text_input_shape, name='text')

	# Pass it through first fully connected layer
	#x = Dense(text_dim, input_shape = text_input_shape)(text_data)

	# first convolutional layer
	x = Convolution1D(128, 3, input_shape = text_input_shape, border_mode = 'same')(text_data)
	x = (Activation('relu'))(x)
	x = (MaxPooling1D(pool_length = 2))(x)

	# Now through the second convolutional layer
	x = (Convolution1D(32, 5, border_mode='same'))(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling1D(pool_length = 2))(x)

	# Flatten our array
	x = Flatten()(x)

	# Define the numerical feature input
	numerical_data = Input(shape=num_input_shape, name='numerical')
	z = Flatten()(numerical_data)

	# Concatenate the output of CNN with the pre-extracted feature input
	concatenated = merge([z, x], mode='concat')

	# Add a fully connected layer
	x = Dense(128, activation='relu')(concatenated)
	x = Dropout(.5)(x)

	# Add a second fully connected layer
	x = Dense(64, activation='relu')(x)

	# Get the final output
	out = Dense(1, activation='softmax')(x)
	# create the model using the Keras functional API
	model = Model(input=[numerical_data, text_data], output=out)
	#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) #can change optimizer settings as desired
	model.compile(loss='mean_squared_error', optimizer='sgd')

	return model

def train_model():
	d = extract.get_data()
	d = extract.clean_data(d)
	d_name_features = extract.build_name_features(d)
	d_state = extract.build_state_features(d, 15)
	d_other = extract.build_other_text_features(d)
	
	d_numeric = pd.concat([d.drop(['state', 'name', 'issuetype', 'issuesource', 'tradetype'], axis = 1), d_state], axis = 1)
	price = d_numeric.pop('price')
	d_text = pd.concat([d_name_features, d_other], axis = 1)
	d = pd.concat([d_numeric, d_text], axis = 1)

	numeric_features = d_numeric.columns.tolist()
	text_features = d_text.columns.tolist()

	#bins = np.array([ 0, 80, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,112.5,115, 120, 150])
	#price = np.digitize(d.pop('price'), bins)


	#scale variables
	#d = StandardScaler().fit_transform(d)

	#split into test and training parts
	d_train, d_test, p_train, p_test = train_test_split(d, price, test_size=0.20, random_state=33)

	d_num_train = d_train.values[:, :len(numeric_features)]
	d_num_test = d_test.values[:, :len(numeric_features)]
	d_text_train = d_train.values[:, len(numeric_features):]
	d_text_test = d_test.values[:, len(numeric_features):]
	p_train = p_train.values
	p_test = p_test.values

	#reshape data appropriately for Keras to use
	if K.image_dim_ordering() == 'th':
		d_num_train = d_num_train.reshape(d_num_train.shape[0], 1, d_num_train.shape[1])
		d_text_train = d_text_train.reshape(d_text_train.shape[0], 1, d_text_train.shape[1])
		d_num_test = d_num_test.reshape(d_num_test.shape[0], 1, d_num_test.shape[1])
		d_text_test = d_text_test.reshape(d_text_test.shape[0], 1, d_text_test.shape[1])
	else:
		d_num_train = d_num_train.reshape(d_num_train.shape[0], d_num_train.shape[1], 1)
		d_text_train = d_text_train.reshape(d_text_train.shape[0], d_text_train.shape[1], 1)
		d_num_test = d_num_test.reshape(d_num_test.shape[0], d_num_test.shape[1], 1)
		d_text_test = d_text_test.reshape(d_text_test.shape[0], d_text_test.shape[1], 1)

	#create model
	model = merged_model(len(numeric_features), len(text_features))

	#train model
	#autosave best model
	best_model_file = "price_predictor_nn.h5"
	best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

	batch_size = 300
	nb_epoch=100

	history = model.fit([d_num_train, d_text_train], p_train,
                              batch_size = batch_size,
                              nb_epoch=nb_epoch,
                              validation_data=([d_num_test, d_text_test], p_test),
                              verbose=0,
                              callbacks=[best_model])

#def predict_test(best_model_file):

#	print('Loading the best model...')
#	model = load_model(best_model_file)
#	print('Best Model loaded!')

#	#compute predictions on test set and create submission
#	#get predictions
#	d_predict_prob = model.predict([image_test_data, num_test_data])

#	# Get the names of the column headers
#	text_labels = sorted(pd.read_csv(num_data_path + "train.csv").species.unique())

#	## Converting the test predictions in a dataframe as depicted by sample submission
#	y_submission = pd.DataFrame(y_predict_prob, index=test_ids, columns=text_labels)

#	fp = open('submission.csv', 'w')
#	fp.write(y_submission.to_csv())	
#	print('Finished writing submission')
#	## Display the submission
#	y_submission.tail()


def main():
	train_model()

if __name__ == "__main__":
	main()

