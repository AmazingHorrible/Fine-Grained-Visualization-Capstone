# -*- coding: utf-8 -*-
"""
Created on May 15 17:56:49 2018

@author: hangz
This script is to do the experiment of comparison between normal transfer learning and transitive transfer learning
Normal TL: train bottleneck feature and fine-tuning
Transitive: do Normal TL for the categories with large image numbers, and then train the model by same dataset with Normal TL.
There are 3 model to select categories: random, similar and non-similar
The difference with NN_driver_Together.py:
    Assume we have two subset for our selected dataset.
    SetLarge is the subset contains categories with image numbers in range A to B
    SetSmall is the subset contains categories with image numbers in range C to D
    And C < D < A < B
    In NN_driver_Together.py, 
    Normal TL will train both SetLarge and SetSmall
    Transitive will train SetLarge first and then, train both SetLarge and SetSmall
    While in this script,
    Normal TL will train only SetSmall
    Transitive will train SetLarge first and then, train SetSmall
"""
import gc
import os.path
import csv
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from readingUlits import readingUlits
from sklearn.cross_validation import train_test_split
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras import backend as K


def plot_training(history, name):
    """
    Draw plot images for testing results
    Args:
        history: training history objects
        name: name of part of experiment, usually: normal, transitive
    """
    plt.ioff()
    acc_path = log_dir + name + '_Accuracy.png'
    loss_path = log_dir + name + '_Loss.png'
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    fig = plt.figure()
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy of ' + name + ' TL')
    plt.savefig(acc_path)
    plt.close(fig)
    fig = plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss of ' + name + 'TL')
    plt.savefig(loss_path)
    plt.close(fig)

def getExperimentId(Id_file):
    """
    Read and update for experiment Id
    Args:
        Id_file: file name that contains current Id
    """
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

def predict(model, img, top, target_size):
    """
    Predict label of an image
    Args:
        model: trained network
        img: image path
        top: top n labels
        target_size: size of image
    """
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x /= 255
    preds = model.predict(x)
    index = np.argmax(preds, axis=top)
    return index

def testing(model, select, filename, error_log, increment):
    """
    Test model by all data with trained labels, save the performance of each categories to file, and update error log for wrong prediction
    Args:
        model: trained model
        select: categories haven been trained
        filename: usually: normal, transitive
        error_log: error log for wrong prediction
        increment: an int value to help to figure out which step the error happened
    """
    cat_performance_path = log_dir + filename + '_categories_performance.csv'
    csv_cols = ['category', 'accuracy']
    if not os.path.isfile(cat_performance_path):
        with open(cat_performance_path, 'a', newline="\n", encoding="utf-8") as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow(csv_cols)
            newFile.close()
    for cat in select:
        imgs = os.listdir(os.getcwd() + '/' + input_file + '/' + cat)
        p = os.getcwd() + '/' + input_file + '/' + cat
        total = 0
        wrong = 0
        for img in imgs:
            if not img.startswith('.'):
                total += 1
                name = img
                img = image.load_img(p + '/' + img, target_size=(299, 299))
                preds = predict(model, img, 1, target_size=(299, 299))
                if cat != selectlabelled[preds[0]]:
                    key = cat + '\t' + name + '\t' + selectlabelled[preds[0]]
                    if key not in error_log:
                        error_log[key] = increment
                    else:
                        error_log[key] += increment
                    wrong += 1
        accuracy = float(total - wrong) / float(total)
        aLine = [cat, "{:.4f}".format(accuracy)]
        with open(cat_performance_path, 'a', newline="\n", encoding="utf-8") as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow(aLine)
            newFile.close()

def errorWriteToFile(error_l, dir_path):
    """
    Write error log to csv file
    Args:
        error_l: error log dictionary
        dir_path: the path of csv file
    """
    print('Writing error log......')
    error_path = dir_path + 'error_log.csv'
    error_cols = ['Category', 'File Name', 'Wrong Prediction', 'Error in']
    with open(error_path, 'w', newline="\n", encoding="utf-8") as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(error_cols)
        for key, value in error_l.items():
            cols = key.split('\t')
            m = ''
            if value == 1:
                m = 'Normal TL'
            elif value == 2:
                m = 'Transitive TL'
            elif value == 3:
                m = 'Both'
            line = [cols[0], cols[1], cols[2], m]
            newFileWriter.writerow(line)
        newFile.close()

#################################
#Start execution
#parameters haven been adjusted, do not change
freeze = 172
b_size = 32
e_cpoch = 60
learning_rate = 0.003
momen = 0.0
opti = 'SGD'
de = 0.0
#################################

#################################
#early stopping point for training
earlyStopping=EarlyStopping(monitor='val_loss', min_delta=0.01, patience = 2, verbose=0, mode='auto')
#################################

#################################
#get experiment id, this id will be increment automatically by 1
Id_file = 'Id2.txt'
Id = getExperimentId(Id_file)
#################################

#################################
#create report directory
report_dir_re = 'report2'
report_dir = r'report2/'
if not os.path.isdir(report_dir_re):
    os.mkdir(report_dir_re)
#################################

#################################
#create log directory for each experiment
log_dir_re = report_dir + "{:d}_".format(Id) + 'log'
log_dir = report_dir + "{:d}_".format(Id) + r'log/'
if not os.path.isdir(log_dir_re):
    os.mkdir(log_dir_re)
#################################

#################################
#create training summary log file
report_file = report_dir + "training_summary_log.csv"
csv_columns = ['Id', 'Input Directory', 'Mode of selecting categories', 
               'Lowerest Image Numbers for step 1', 'Highest Image Numbers for step 1', 'Number of categories for step 1',
               'Lowerest Image Numbers for step 2', 'Highest Image Numbers for step 2', 'Number of categories for step 2', 
               'Validation Loss of normal TL', 'Validation Accuracy of normal TL', 'Execution Time of normal TL',
               'Validation Loss of transitive TL', 'Validation Accuracy of transitive TL', 'Execution Time of transitive TL']
if not os.path.isfile(report_file):
    with open(report_file, 'w', newline="\n", encoding="utf-8") as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(csv_columns)
        newFile.close()
#################################

#################################
#The section you may edit during the experiment
#(To selection the sub categories you want to train, basically depends on the number of images)
mode = 1 # 1: randomly pick categories, 2: pick categories with similar names, 3: pick categories with non-similar names
input_file = 'Insecta' # which super category you want to train
rangeMin = 50 # the categories for step 1 which at least x(number) of images
rangeMax = 100 # the categories for step 1 which at most x(number) of images
number_of_cats = 10 # How many categories you wan to train for seep 1
rangeMin2 = 0 # the categories for step 2 which at least x(number) of images
rangeMax2 = 50 # the categories for step 2 which at most x(number) of images
number_of_cats2 = 10 # How many categories you wan to train for step 2
#################################

#################################
#functions for select categories and read image files: 
#setCats, setCatsSimilar, setCatsNonSimilar
#appendCats, appendCatsSimilar, appendCatsNonSimilar
reader = readingUlits(input_file) #initialize readingUlits object
if mode == 1:
    reader.setCats(rangeMin, rangeMax, number_of_cats)
elif mode == 2:
    reader.setCatsSimilar(rangeMin, rangeMax, number_of_cats)
else:
    reader.setCatsNonSimilar(rangeMin, rangeMax, number_of_cats)
select = reader.getSelect() #save the selected categories without label
images = reader.setInfo()
selectlabelled = reader.getLabelledInfo() #save the selected categories with label
print(selectlabelled)
#################################

#################################
#setup labels for data
label_name = reader.getLabelNames()
locations = reader.getLocations()
number_of_classess = reader.getSubCategoryNumber()
labels = np.ones((reader.getTotalImageNumber(),),dtype='int64')
prev = 0
for i in range(number_of_classess):
    labels[prev:prev + locations[i]] = i
    prev += locations[i]
res = np_utils.to_categorical(labels, number_of_classess)
#split dataset randomly into two subset: training dataset and testing dataset
X_train, X_test, y_train, y_test = train_test_split(images, res, test_size=0.1, random_state=2)
#Data agumentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(X_train)
#################################

#################################
#start time of normal TL
start_time = time.time()
#################################

#################################
#load model
image_input = Input(shape=(299, 299, 3))
base_model = InceptionV3(input_tensor=image_input, weights='imagenet', include_top=False)
#add dense layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(number_of_classess, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#freeze all layers for training dense layer
for layer in base_model.layers:
    layer.trainable = False
#training dense layer
history_ft = model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#################################

#################################
print('Training (Phase 1)------------')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, 
                    epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
#################################

#################################
#freeze some layers and fine-tuning
for layer in model.layers[:freeze]:
   layer.trainable = False
for layer in model.layers[freeze:]:
   layer.trainable = True
print('Fine-tuning (Phase 1)------------')
if opti == 'SGD':
    model.compile(optimizer=SGD(lr=learning_rate, momentum=momen, decay = de), loss='categorical_crossentropy', metrics=['accuracy'])
elif opti == 'RMSprop':
    model.compile(optimizer=RMSprop(lr=learning_rate, decay=de), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, 
                                 epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
#Testing with testing dataset
(loss_normal, accuracy_normal) = model.evaluate(X_test, y_test, batch_size=b_size, verbose=1)
#################################

#################################
#save plots for normal TL
plot_training(history_ft, 'Normal')
#################################

#################################
#Testing with all data (including both training and testing dataset)
#Then save the accuracy for each category and record all error data records
error_log = {}
testing(model, select, 'Normal', error_log, 1)
#################################

#################################
#record execution time for normal TL
end_time = time.time()
interval = end_time - start_time
td_normal = str(datetime.timedelta(seconds=interval))
#################################

#################################
#delete normal TL model and release memory
K.clear_session() #to avoid clutter from old models / layers.
sess = tf.Session()
K.set_session(sess)
del base_model
del model
gc.collect()
#################################

#################################
#TTL
#functions for select categories and read image files: 
#setCats, setCatsSimilar, setCatsNonSimilar
#appendCats, appendCatsSimilar, appendCatsNonSimilar
reader1 = readingUlits(input_file) #initialize readingUlits object
if mode == 1:
    reader1.setCats(rangeMin2, rangeMax2, number_of_cats2)
elif mode == 2:
    reader1.setCatsSimilar(rangeMin, rangeMax, number_of_cats)
    reader1.setCatsSimilarForReplace(rangeMin2, rangeMax2, number_of_cats2)
else:
    reader1.setCatsNonSimilar(rangeMin, rangeMax, number_of_cats)
    reader1.setCatsNonSimilarForReplace(rangeMin2, rangeMax2, number_of_cats2)
images1 = reader1.setInfo()
#################################

#################################
#setup labels for data
label_name1 = reader1.getLabelNames()
locations1 = reader1.getLocations()
number_of_classess1 = reader1.getSubCategoryNumber()
labels1 = np.ones((reader1.getTotalImageNumber(),),dtype='int64')
prev1 = 0
for i in range(number_of_classess1):
    labels1[prev1:prev1 + locations1[i]] = i
    prev1 += locations1[i]
res1 = np_utils.to_categorical(labels1, number_of_classess1)
#split dataset randomly into two subset: training dataset and testing dataset
X_train1, X_test1, y_train1, y_test1 = train_test_split(images1, res1, test_size=0.1, random_state=2)
#Data agumentation
datagen.fit(X_train1)
#################################

#################################
#start time of transitive TL
start_time = time.time()
#################################

#################################
#TTL Phase 1
#load model
image_input1 = Input(shape=(299, 299, 3))
base_model1 = InceptionV3(input_tensor=image_input1, weights='imagenet', include_top=False)
#add dense layers
x1 = base_model1.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dense(1024, activation='relu')(x1)
predictions1 = Dense(number_of_classess1, activation='softmax')(x1)
model1 = Model(inputs=base_model1.input, outputs=predictions1)
#freeze all layers for training dense layer
for layer in base_model1.layers:
    layer.trainable = False
#training dense layer
history_ft_temp = model1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
print('Training (Phase 2: step 1) ------------')
model1.fit_generator(datagen.flow(X_train1, y_train1, batch_size=b_size), steps_per_epoch=len(X_train1) / b_size, 
                    epochs=e_cpoch, verbose=1, validation_data=(X_test1, y_test1), callbacks=[earlyStopping], shuffle=True)
for layer in model1.layers[:freeze]:
   layer.trainable = False
for layer in model1.layers[freeze:]:
   layer.trainable = True
print('Fine-tuning (Phase 2: step 1)------------')
if opti == 'SGD':
    model1.compile(optimizer=SGD(lr=learning_rate, momentum=momen, decay = de), loss='categorical_crossentropy', metrics=['accuracy'])
elif opti == 'RMSprop':
    model1.compile(optimizer=RMSprop(lr=learning_rate, decay=de), loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit_generator(datagen.flow(X_train1, y_train1, batch_size=b_size), steps_per_epoch=len(X_train1) / b_size, 
                    epochs=e_cpoch, verbose=1, validation_data=(X_test1, y_test1), callbacks=[earlyStopping], shuffle=True)
#Release memory for reuse
del images1
del X_train1
del y_train1
del X_test1
del y_test1
gc.collect()
#################################

#################################
#TTL phase 2
#Remove previous dense layer
model1.layers.pop()
x = model1.layers[-1].output
predictions = Dense(number_of_classess, activation='softmax')(x)
model1 = Model(inputs=base_model1.input, outputs=predictions)
for layer in base_model1.layers:
    layer.trainable = False
history_ft = model1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
print('Training (Phase 2: step 2)------------')
model1.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, 
                    epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
for layer in model1.layers[:freeze]:
   layer.trainable = False
for layer in model1.layers[freeze:]:
   layer.trainable = True
print('Fine-tuning (Phase 2: step 2) ------------')
if opti == 'SGD':
    model1.compile(optimizer=SGD(lr=learning_rate, momentum=momen, decay = de), loss='categorical_crossentropy', metrics=['accuracy'])
elif opti == 'RMSprop':
    model1.compile(optimizer=RMSprop(lr=learning_rate, decay=de), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft1 = model1.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, 
                                  epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
#Testing with testing dataset
(loss_transitive, accuracy_transitive) = model1.evaluate(X_test, y_test, batch_size=b_size, verbose=1)
#################################

#################################
#save plots for transitive TL
plot_training(history_ft1, 'Transitive')
#################################

#################################
#Testing with all data (including both training and testing dataset)
#Then save the accuracy for each category and record all error data records
#Write all error log to file
testing(model1, select, 'Transitive', error_log, 2)
errorWriteToFile(error_log, log_dir)
#################################

#################################
#record execution time for transitive TL
end_time = time.time()
interval = end_time - start_time
td_transitive = str(datetime.timedelta(seconds=interval))
#################################

#################################
#Writing overall summary
m = ''
if mode == 1:
    m = 'random'
elif mode == 2:
    m = 'similar'
elif mode == 3:
    m = 'non similar'
summarys = ["{:d}".format(Id), input_file, m, 
               "{:d}".format(rangeMin), "{:d}".format(rangeMax), "{:d}".format(number_of_cats),
               "{:d}".format(rangeMin2), "{:d}".format(rangeMax2), "{:d}".format(number_of_cats2), 
               "{:.4f}".format(loss_normal), "{:.4f}".format(accuracy_normal), td_normal,
               "{:.4f}".format(loss_transitive), "{:.4f}".format(accuracy_transitive), td_transitive]
with open(report_file, 'a', newline="\n", encoding="utf-8") as newFile:
    newFileWriter = csv.writer(newFile)
    newFileWriter.writerow(summarys)
    newFile.close()
#################################

#################################
#Free all memory occupied by this program
del base_model1
del model1
del images
del X_train
del y_train
del X_test
del y_test
gc.collect()
#################################

#Finihsed
print('Successfully done.')


