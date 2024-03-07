import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import imageio
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from numpy.random import seed
seed(101)
tf.random.set_seed(101)

#print(tf.config.experimental.list_physical_devices('GPU'))

all_images_dir = 'all_images_dir'
'''
# Create a new directory to store all available images

os.mkdir(all_images_dir)

path_0 = "D://desktop/dataset/0"
path_1 = "D://desktop/dataset/1"
# create a list of all files in folder 0
file_list_0 = os.listdir(path_0)
# create a list of list all file in folder 1
file_list_1 = os.listdir(path_1)

# move the 0 images to all_images_dir
for fname in file_list_0:
    # source path to image
    src = os.path.join(path_0, fname)
    # destination path to image
    dst = os.path.join(all_images_dir, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

# move the 1 images to all_images_dir
for fname in file_list_1:

    # source path to image
    src = os.path.join(path_1, fname)
    # destination path to image
    dst = os.path.join(all_images_dir, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)
    

#print(len(os.listdir('all_images_dir')))

image_list = os.listdir('all_images_dir')

df_data = pd.DataFrame(image_list, columns=['image_id'])
#print(df_data.head())

##Function to extract target
def extract_target(x):
    a = x.split('_')
    # target = index 4
    b = a[4]
    #print(b)
    # the ytarget i.e. 1 or 2 is the 5th index of the string --> class1
    target = b[5]
    return target

x = df_data
df_data['target'] = df_data['image_id'].apply(extract_target)
#print(df_data.head())
print(df_data['target'].value_counts())

#TRAIN-TEST SPLIT
y = df_data['target']
df_train, df_test = train_test_split(df_data, test_size=0.3,train_size=0.7, random_state=101, stratify=y)
print(df_train['target'].value_counts())
print(df_test['target'].value_counts())


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)


#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 2 folders inside 'base_dir':
# create a path to 'base_dir' to which we will join the names of the new folders
# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# test_dir
test_dir = os.path.join(base_dir, 'test_dir')
os.mkdir(test_dir)

# create new folders inside train_dir
no_idc = os.path.join(train_dir, 'no_idc')
os.mkdir(no_idc)
idc = os.path.join(train_dir, 'idc')
os.mkdir(idc)

# create new folders inside test_dir
no_idc = os.path.join(test_dir, 'no_idc')
os.mkdir(no_idc)
idc = os.path.join(test_dir, 'idc')
os.mkdir(idc)
# check that the folders have been created
print(os.listdir('base_dir/train_dir'))
print(os.listdir('base_dir/test_dir'))


# Set the id as the index in df_data
df_data.set_index('image_id', inplace=True)
train_list = list(df_train['image_id'])
test_list = list(df_test['image_id'])

# TRANSFER TRAINING AND TESTING IMAGES
#train_dir = 'base_dir/train_dir'
#test_dir = 'base_dir/test_dir'
for image in train_list:

    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image
    # get the label for a certain image
    target = df_data.loc[image, 'target']

    # these must match the folder names
    if target == '0':
        label = 'no_idc'
    if target == '1':
        label = 'idc'

    # source path to image
    src = os.path.join(all_images_dir, fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # move the image from the source to the destination
    shutil.move(src, dst)

# Transfer the test images

for image in test_list:

    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image
    # get the label for a certain image
    target = df_data.loc[image, 'target']

    # these must match the folder names
    if target == '0':
        label = 'no_idc'
    if target == '1':
        label = 'idc'

    # source path to image
    src = os.path.join(all_images_dir, fname)
    # destination path to image
    dst = os.path.join(test_dir, label, fname)
    # move the image from the source to the destination
    shutil.move(src,dst)

# check how many train images we have in each folder

print(len(os.listdir('base_dir/train_dir/no_idc')))
print(len(os.listdir('base_dir/train_dir/idc')))
'''

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/test_dir'


num_train_samples = 1400
num_val_samples = 600
train_batch_size = 10
val_batch_size = 10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
IMAGE_SIZE = 50
datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
                 input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

model.compile(Adam(lr=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              epochs=60, verbose=1,
                              callbacks=callbacks_list)

# # get the metric names so we can use evaulate_generator
# print(model.metrics_names)
# # Here the best epoch will be used.
#
# model.load_weights('model.h5')
#
# val_loss, val_acc = model.evaluate_generator(test_gen, steps=600)
#
# print('val_loss:', val_loss)
# print('val_acc:', val_acc)

# make a prediction
# predictions = model.predict_generator(test_gen, steps=600, verbose=1)
# print(predictions.shape)
# # This is how to check what index keras has internally assigned to each class.
# print(test_gen.class_indices)
# Put the predictions into a dataframe.
# The columns need to be ordered to match the output of the previous cell

# df_preds = pd.DataFrame(predictions, columns=['idc', 'no_idc'])
#
# print(df_preds.head())
#
# # Get the true labels
# y_true = test_gen.classes
#
# # Get the predicted labels as probabilities
# y_pred = df_preds['no_idc']
# from sklearn.metrics import roc_auc_score
#
# print(roc_auc_score(y_true, y_pred))


#Save the model

# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
