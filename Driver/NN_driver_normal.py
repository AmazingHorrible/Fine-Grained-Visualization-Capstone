# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:56:49 2018

@author: hangz
"""
import numpy as np
from keras.utils import np_utils
from readingUlits import readingUlits
from sklearn.cross_validation import train_test_split
from keras.applications.inception_v3 import InceptionV3,  decode_predictions
from keras.models import Model
from keras.layers import merge, Input, GlobalAveragePooling2D, Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import gc
import os.path
import csv
import time
import datetime
from keras.preprocessing.image import ImageDataGenerator


def plot_training(history, Id):
    if os.path.isdir(plots_re):
        acc_path = plots_dir + str(Id) + '_Accuracy.png'
        loss_path = plots_dir + str(Id) + '_Loss.png'
    else:
        os.mkdir(plots_re)
        acc_path = plots_dir + str(Id) + '_Accuracy.png'
        loss_path = plots_dir + str(Id) + '_Loss.png'
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig(acc_path)
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig(loss_path)
    plt.show()

def getExperimentId(Id_file):
    rev = 0
    if os.path.isfile(Id_file):
        file = open(Id_file, 'r')
        revStr =  file.readline()
        rev = int(revStr)
        rev += 1
        file.close()
        file = open(Id_file,'w')
        file.write(str(rev))
        file.close()
    else:
        file = open(Id_file,'w')
        file.write('0')
        file.close()
    return rev



report_dir_re = 'normal'
report_dir = r'normal/'
plots_re = report_dir + 'plots'
plots_dir = report_dir + r'plots/'
report_file = report_dir + "training_log.csv"
Id_file = report_dir + 'Id.txt'
input_file = 'Mammalia'

start_time = time.time()
csv_columns = ['Id', 'Input Directory', 'Lowerest Image Numbers for large', 'Highest Image Numbers for large', 'Number of categories for large' , 'Lowerest Image Numbers for small', 'Highest Image Numbers for small', 'Number of categories for small', 'Validation Loss', 'Validation Accuracy', 'Execution Time']


#################################
#The section you may edit during the experiment
#(To selection the sub categories you want to train, basically depends on the number of images)
cat_extract_option = 1 #the range mode for category selection: a < x < b
rangeMin = 500 # a = 500
rangeMax = 1000 # b = 1000
number_of_cats = 12 # How many categories you wan to train

cat_extract_option2 = 3 # the range mode for category selection: x < a
rangeMin2 = 50 # a = 50
rangeMax2 = 0 # useless for this mode, just for writing to file purpose
number_of_cats2 = 12 # How many categories you wan to train
#################################

freeze = 172
b_size = 32
e_cpoch = 60
learning_rate = 0.003
momen = 0.0

opti = 'SGD'
de = 0.0

if not os.path.isdir(report_dir_re):
    os.mkdir(report_dir_re)

Id = getExperimentId(Id_file)
reader = readingUlits(input_file)
output_file_name = ""

reader.setCats(cat_extract_option, rangeMin, rangeMax, number_of_cats)
reader.appendCats(cat_extract_option2, rangeMin2, rangeMax2, number_of_cats2)

images = reader.setInfo()

label_name = reader.getLabelNames()
locations = reader.getLocations()
number_of_classess = reader.getSubCategoryNumber()
labels = np.ones((reader.getTotalImageNumber(),),dtype='int64')
prev = 0

for i in range(number_of_classess):
    labels[prev:prev + locations[i]] = i
    prev += locations[i]

res = np_utils.to_categorical(labels, number_of_classess)

X_train, X_test, y_train, y_test = train_test_split(images, res, test_size=0.1, random_state=2)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(X_train)

earlyStopping=EarlyStopping(monitor='val_acc', min_delta=0.00001, patience = 5, verbose=0, mode='max')

image_input = Input(shape=(299, 299, 3))

base_model = InceptionV3(input_tensor=image_input, weights='imagenet', include_top=False)



x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(number_of_classess, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
#model.summary()
history_ft = model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training ------------')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
#plot_training(history_ft)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:freeze]:
   layer.trainable = False
for layer in model.layers[freeze:]:
   layer.trainable = True
print('Fine-tuning ------------')
if opti == 'SGD':
    model.compile(optimizer=SGD(lr=learning_rate, momentum=momen, decay = de), loss='categorical_crossentropy', metrics=['accuracy'])
elif opti == 'RMSprop':
    model.compile(optimizer=RMSprop(lr=learning_rate, decay=de), loss='categorical_crossentropy', metrics=['accuracy'])

#model.summary()

history_ft = model.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)

plot_training(history_ft, Id)
print('\nTesting ------------')
(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=b_size, verbose=1)
print("loss={:.4f}, accuracy: {:.4f}".format(loss,accuracy))

end_time = time.time()
interval = end_time - start_time
td = str(datetime.timedelta(seconds=interval))

summarys = ["{:d}".format(Id), input_file, "{:d}".format(rangeMin), "{:d}".format(rangeMax), "{:d}".format(number_of_cats),"{:d}".format(rangeMin2), "{:d}".format(rangeMax2), "{:d}".format(number_of_cats2),  "{:.4f}".format(loss), "{:.4f}".format(accuracy), td]
if os.path.isfile(report_file):
    with open(report_file, 'a', newline="\n", encoding="utf-8") as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(summarys)
        newFile.close()
else:
    with open(report_file,'w', newline="\n", encoding="utf-8") as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(csv_columns)
        newFileWriter.writerow(summarys)
        newFile.close()

del images
del X_train
del y_train
del X_test
del y_test
gc.collect()

#history_ft = model.fit(X_train, y_train, batch_size=b_size, epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
"""

"""


