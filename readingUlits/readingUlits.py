# -*- coding: utf-8 -*-
"""
Functions for reading testing files, receiving and reading results and statistics

@author: shuangm
"""
import os
import numpy as np
from keras.preprocessing import image
from statistic import staticAnalysis
from keras.applications.imagenet_utils import preprocess_input

class readingUlits:
    path = ''
    rate = 0.0
    filenamelist = []
    filefullnamelist = []
    images = []
    label_names = []
    locations = []
    sub_category_number = 0
    total_image_number = 0
    
    labeled = {}
    select = []

    def __init__(self, folderPath):
        self.path = folderPath
       ######################################################################## 
       #The functions in this block are quite similar
       #The purpose of these funtion is to select categories and save it in self.select
       #There are three mode
       #To inspect usage of these mode please check driver script
    def setCats(self, small, large, num):
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select = analyzer.rangeCategories(small, large, num)
        else:
            self.select = analyzer.rangeCategories(small, large, 0)
        
    def appendCats(self, small, large, num):    
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select.extend(analyzer.rangeCategories(small, large, num))
        else:
            self.select.extend(analyzer.rangeCategories(small, large, 0))
            
    def setCatsSimilar(self, small, large, num):
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select = analyzer.rangeCategoriesSimilar(small, large, num)
        else:
            self.select = analyzer.rangeCategoriesSimilar(small, large, 0)
        
    def setCatsSimilarForReplace(self, small, large, num):    
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select = analyzer.rangeCategoriesSimilarTo(small, large, num, self.select)
        else:
            self.select = analyzer.rangeCategoriesSimilarTo(small, large, 0, self.select)
        
    def appendCatsSimilar(self, small, large, num):    
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select.extend(analyzer.rangeCategoriesSimilarTo(small, large, num, self.select))
        else:
            self.select.extend(analyzer.rangeCategoriesSimilarTo(small, large, 0, self.select))
            
    def setCatsNonSimilar(self, small, large, num):
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select = analyzer.rangeCategoriesNonSimilar(small, large, num)
        else:
            self.select = analyzer.rangeCategoriesNonSimilar(small, large, 0)
        
    def setCatsNonSimilarForReplace(self, small, large, num):    
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select = analyzer.rangeCategoriesNonSimilarTo(small, large, num, self.select)
        else:
            self.select = analyzer.rangeCategoriesNonSimilarTo(small, large, 0, self.select)
        
    def appendCatsNonSimilar(self, small, large, num):    
        analyzer = staticAnalysis(self.path)
        analyzer.statisticSummary()
        if (num != 0):
            self.select.extend(analyzer.rangeCategoriesNonSimilarTo(small, large, num, self.select))
        else:
            self.select.extend(analyzer.rangeCategoriesNonSimilarTo(small, large, 0, self.select))
        #######################################################################
        
        #######################################################################
        #Some get functions to return necessary data fields
    def getLabelNames(self):
        return self.label_names
    
    def getLocations(self):
        return self.locations
    
    def getSubCategoryNumber(self):
        return self.sub_category_number
    
    def getTotalImageNumber(self):
        return self.total_image_number
    
    def getImages(self):
        return self.images
    
    def getSelect(self):
        return self.select
    
    def getLabelledInfo(self):
        rev = {val:key for (key, val) in self.labeled.items()}
        return rev
        #######################################################################
    
    def setInfo(self):
        """
        This function will read all image data and return.
        During the process, all necessary information will also be set
        For example,
        Number of images
        Number of categories
        Labels of images ... etc ...
        """
        num_of_loaded = 0
        sub_category_list = os.listdir(os.getcwd() + '/' + self.path)
        image_list = []
        self.sub_category_number = 0
        self.locations = []
        self.total_image_number = 0
        for sub_category in sub_category_list:
            if sub_category in self.select:
                self.sub_category_number += 1
            else:
                continue
            imgs = os.listdir(os.getcwd() + '/' + self.path + '/' + sub_category)
            p = os.getcwd() + '/' + self.path + '/' + sub_category
            self.label_names.append(sub_category)
            image_num = 0
            for img in imgs:
                if not img.startswith('.'):
                    img = image.load_img(p + '/' + img, target_size=(299, 299))
                    temp = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0)) / 255
                    image_list.append(temp)
                    self.total_image_number += 1
                    image_num += 1
            self.locations.append(image_num)
            print("{:d} images of category  < {:s} > have been loaded.".format(image_num, sub_category))
            self.labeled[sub_category] = num_of_loaded
            num_of_loaded += 1
            print("{:d} sub-categories has been loaded......(total {:d})".format(num_of_loaded, self.total_image_number))
        self.images = np.rollaxis(np.array(image_list), 1, 0)
        self.images=self.images[0]
        self.select = []
        return self.images
         
        #######################################################################  
        #Test functions for find images 
    def getFullFileNames(self):
        for root, dirs, files in os.walk(self.path):
            for name in files:
                fullname = os.path.join(root, name)
                folder, subfolder = os.path.split(root)
                tupleTMP =  (fullname, folder + "->" + subfolder)
                self.filefullnamelist.append(tupleTMP)
        return self.filefullnamelist
    
    def getFileNames(self):
        for root, dirs, files in os.walk(self.path):
            for name in files:
                fullname = os.path.join('', name)
                folder, subfolder = os.path.split(root)
                tupleTMP =  (fullname, folder + "->" + subfolder)
                self.filenamelist.append(tupleTMP)
        return self.filenamelist
            
    def getFullSize(self):
        return len(self.filefullnamelist)
            
    def getSize(self):
        return len(self.filenamelist)
        #######################################################################
