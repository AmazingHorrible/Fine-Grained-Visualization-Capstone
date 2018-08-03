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
import argparse
import GPUtil


def creatList(label, loss, acc, count):
    rev = []
    rev.append(label)
    rev.append(loss)
    rev.append(acc)
    rev.append(count)
    return rev

def appendList(l, loss, acc, count):
    cols = [1, 2, 4, 5, 7]
    l.append(loss)
    l.append(acc)
    l.append(count)
    l.append(acc - l[2])
    for c in cols:
        l[c] = "{:.4f}".format(l[c])
    return l
#=normal_and_transitive
def plot_training(historyN, historyT):
    """
    Draw plot images for testing results
    Args:
        history: training history objects
        name: name of part of experiment, usually: normal, transitive
    """
    plt.ioff()
    acc_path = log_dir + 'Accuracy.png'
    accN = historyN.history['acc']
    val_accN = historyN.history['val_acc']
    accT = historyT.history['acc']
    val_accT = historyT.history['val_acc']
    fig = plt.figure()
    plt.plot(range(len(accN)), val_accN, color="r", linestyle="-", linewidth=1, label="Normal_val_acc")
    plt.plot(range(len(accN)), accN, color="r", linestyle=":", marker="^", linewidth=1, label="Normal_acc")
    plt.plot(range(len(accT)), val_accT, color="b", linestyle="-", linewidth=1, label="Transitive_val_acc")
    plt.plot(range(len(accT)), accT, color="b", linestyle=":", marker="^", linewidth=1, label="Transitive_acc")
    plt.title('Training and validation accuracy of TL')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(acc_path)
    plt.close(fig)
    
    plt.ioff()
    loss_path = log_dir + 'Loss.png'
    lossN = historyN.history['loss']
    val_lossN = historyN.history['val_loss']
    lossT = historyT.history['loss']
    val_lossT = historyT.history['val_loss']
    fig = plt.figure()
    plt.plot(range(len(lossN)), val_lossN, color="r", linestyle="-", linewidth=1, label="Normal_val_loss")
    plt.plot(range(len(lossN)), lossN, color="r", linestyle=":", marker="^", linewidth=1, label="Normal_loss")
    plt.plot(range(len(lossT)), val_lossT, color="b", linestyle="-", linewidth=1, label="Transitive_val_loss")
    plt.plot(range(len(lossT)), lossT, color="b", linestyle=":", marker="^", linewidth=1, label="Transitive_loss")
    plt.title('Training and validation loss of TL')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
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
    print('Recording errors')
    for cat in select:
        imgs = os.listdir(os.getcwd() + '/' + input_file + '/' + cat)
        p = os.getcwd() + '/' + input_file + '/' + cat
        for img in imgs:
            if not img.startswith('.'):
                name = img
                img = image.load_img(p + '/' + img, target_size=(299, 299))
                preds = predict(model, img, 1, target_size=(299, 299))
                if cat != selectlabelled[preds[0]][0]:
                    key = cat + '\t' + name + '\t' + selectlabelled[preds[0]][0]
                    if key not in error_log:
                        error_log[key] = increment
                    else:
                        error_log[key] += increment

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
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="the directory of input files", default='Insecta')
parser.add_argument("--mode", help="mode of selecting categories", default='1')
parser.add_argument("--tMin", help="lower bound for target categories image numbers", default='30')
parser.add_argument("--tMax", help="upper bound for target categories image numbers", default='50')
parser.add_argument("--tNum", help="number of target categories", default='50')
parser.add_argument("--iMin", help="lower bound for intermediate categories image numbers", default='300')
parser.add_argument("--iMax", help="upper bound for intermediate categories image numbers", default='500')
parser.add_argument("--iNum", help="number of intermediate categories", default='10')
parser.add_argument("--freeze", help="number of layer to freeze", default='172')
parser.add_argument("--bsize", help="batach size", default='32')
parser.add_argument("--epoch", help="number of epoches", default='60')
parser.add_argument("--lr", help="learning rate", default='0.05')
parser.add_argument("--momen", help="momen", default='0.0')
parser.add_argument("--optimizer", help="optimizer function", default='SGD')
parser.add_argument("--decay", help="decay", default='0.0')
parser.add_argument("--delta", help="delta for early stopping", default='0.005')
parser.add_argument("--patience", help="patience for early stopping", default='3')
#################################
    
#################################
#Start execution
#parameters haven been adjusted, do not change
args = parser.parse_args()
freeze = int(args.freeze)
b_size = int(args.bsize)
e_cpoch = int(args.epoch)
learning_rate = float(args.lr)
momen = float(args.momen)
opti = args.optimizer
de = float(args.decay)
#################################

#################################
#early stopping point for training
earlyStopping=EarlyStopping(monitor='val_loss', min_delta=float(args.delta), patience = int(args.patience), verbose=0, mode='auto')
#################################

#################################
#get experiment id, this id will be increment automatically by 1
input_file = args.dir # which super category you want to train
Id_file = input_file + '_IdTLImba.txt'
Id = getExperimentId(Id_file)
#################################

#################################
#create report directory
report_dir_re = input_file +'_reportTLImba'
report_dir = input_file+ r'_reportTLImba/'
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
               'Validation Loss of transitive TL', 'Validation Accuracy of transitive TL', 'Execution Time of transitive TL', 'Increase']
if not os.path.isfile(report_file):
    with open(report_file, 'w', newline="\n", encoding="utf-8") as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(csv_columns)
        newFile.close()
#################################

#################################
#The section you may edit during the experiment
#(To selection the sub categories you want to train, basically depends on the number of images)
mode = int(args.mode) # 1: randomly pick categories, 2: pick categories with similar names, 3: pick categories with non-similar names
rangeMin = int(args.tMin) # the categories for step 1 which at least x(number) of images
rangeMax = int(args.tMax) # the categories for step 1 which at most x(number) of images
number_of_cats = int(args.tNum) # How many categories you wan to train for seep 1
rangeMin2 = int(args.iMin) # the categories for step 2 which at least x(number) of images
rangeMax2 = int(args.iMax) # the categories for step 2 which at most x(number) of images
number_of_cats2 = int(args.iNum) # How many categories you wan to train for step 2
#################################

#################################
#functions for select categories and read image files: 
#setCats, setCatsSimilar, setCatsNonSimilar
#appendCats, appendCatsSimilar, appendCatsNonSimilar
reader = readingUlits(input_file) #initialize readingUlits object
if mode == 1:
    reader.setCats(rangeMin, rangeMax, number_of_cats)
    reader.appendCats(rangeMin2, rangeMax2, number_of_cats2)
elif mode == 2:
    reader.setCatsSimilar(rangeMin, rangeMax, number_of_cats)
    reader.appendCatsSimilar(rangeMin2, rangeMax2, number_of_cats2)
else:
    reader.setCatsNonSimilar(rangeMin, rangeMax, number_of_cats)
    reader.appendCatsNonSimilar(rangeMin2, rangeMax2, number_of_cats2)
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
#GPUtil.showUtilization(all=True, attrList=None, useOldCode=False)
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
history_ft_normal = model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
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
history_ft_normal = model.fit_generator(datagen.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train) / b_size, 
                                 epochs=e_cpoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping], shuffle=True)
#Testing with testing dataset
(loss_normal, accuracy_normal) = model.evaluate(X_test, y_test, batch_size=b_size, verbose=1)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(loss_normal, accuracy_normal))
#################################
print('---------------------------------\nTesting for each label')
for k in range(number_of_classess):
    temp_test = []
    temp_train = []
    count = 0
    for i in range(int(y_test.size/number_of_classess)):
        if y_test[i][k] == 1:
            count += 1
            temp_test.extend(y_test[i].tolist())
            temp_train.extend(X_test[i].tolist())
    if (count == 0):
        selectlabelled[k] = creatList(selectlabelled[k], 0.0, 0.0, 0)
        continue
    temp_test = np.array(temp_test)
    temp_test = np.reshape(temp_test, (count, number_of_classess))
    temp_train = np.array(temp_train)
    temp_train = np.reshape(temp_train, (count, 299, 299, 3))
    temp_loss, temp_acc = model.evaluate(temp_train, temp_test, batch_size=b_size, verbose=1)
    selectlabelled[k] = creatList(selectlabelled[k], temp_loss, temp_acc, count)

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
#GPUtil.showUtilization(all=True, attrList=None, useOldCode=False)
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
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(loss_transitive, accuracy_transitive))
#################################

print('---------------------------------\nTesting for each label')
for k in range(number_of_classess):
    temp_test = []
    temp_train = []
    count = 0
    for i in range(int(y_test.size/number_of_classess)):
        if y_test[i][k] == 1:
            count += 1
            temp_test.extend(y_test[i].tolist())
            temp_train.extend(X_test[i].tolist())
    if (count == 0):
        selectlabelled[k] = appendList(selectlabelled[k], 0.0, 0.0, 0)
        continue
    temp_test = np.array(temp_test)
    temp_test = np.reshape(temp_test, (count, number_of_classess))
    temp_train = np.array(temp_train)
    temp_train = np.reshape(temp_train, (count, 299, 299, 3))
    temp_loss, temp_acc = model1.evaluate(temp_train, temp_test, batch_size=b_size, verbose=1)
    selectlabelled[k] = appendList(selectlabelled[k], temp_loss, temp_acc, count)

cat_performance_path = log_dir + 'categories_testing_performance.csv'
csv_cols = ['category','N_loss', 'N_accuracy', 'N_count' ,'T_loss', 'T_accuracy', 'T_count', 'Increase']
with open(cat_performance_path, 'w', newline="\n", encoding="utf-8") as newFile:
    newFileWriter = csv.writer(newFile)
    newFileWriter.writerow(csv_cols)
    for i in range(number_of_classess):
        newFileWriter.writerow(selectlabelled[i])
    newFile.close()
#################################
#save plots for transitive TL
plot_training(history_ft_normal, history_ft1)
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
               "{:.4f}".format(loss_transitive), "{:.4f}".format(accuracy_transitive), td_transitive, "{:.4f}".format(accuracy_transitive - accuracy_normal)]
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


