# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:58:34 2018

@author: shuangm
Produce statistic summary for super categories and help to select categories
"""
import os
import numpy as np
import random

class staticAnalysis:
    
    path = ''
    sub_categories = []
    sub_numbers = []
    
    total_image = 0
    mean = 0.0
    std = 0.0
    var = 0.0
    num_cat = 0
    
    MAX = 0
    MIN = 100000000
    
    def __init__(self, folderPath):
        self.path = folderPath

    def statisticSummary(self):
        """
        statistic Summary for image numbers of categories
        """
        self.sub_categories = []
        self.sub_numbers = []
        sub_category_list = os.listdir(os.getcwd() + '/' + self.path)
        for sub_category in sub_category_list:
            imgs = os.listdir(os.getcwd() + '/' + self.path + '/' + sub_category)
            image_num = 0
            for img in imgs:
                if not img.startswith('.'):
                    image_num += 1
            self.sub_categories.append(sub_category)
            self.sub_numbers.append(image_num)
            if image_num > self.MAX:
                self.MAX = int(image_num)
            if image_num < self.MIN:
                self.MIN = int(image_num)
        x = np.array(self.sub_numbers)
        self.total_image = np.sum(x)
        self.mean = np.mean(x)
        self.var = np.var(x)
        self.std = np.std(x)
        self.num_cat = len(self.sub_numbers)

    def printSum(self):
        """
        Print summary
        """
        print("Total image numbers: {:d}".format(self.total_image))
        print("In {:d} categories.".format(self.num_cat))
        print("Mean: {:f}".format(self.mean))
        print("Var: {:f}".format(self.std))
        print("Std: {:f}".format(self.var))
        print("MAX: {:d}".format(self.MAX))
        print("MIN: {:d}".format(self.MIN))
            
    def printNum(self):
        """
        Print total number of images
        """
        for i in range(len(self.sub_categories)):
            print("{:s} have: {:d} images".format(self.sub_categories[i], self.sub_numbers[i]))            
    
    def rangeCategories(self, images_num_small, images_num_large, categories_num):
        """
        For giving range of image numbers, and number of categories wanted, randomly pick a list of category names and return
        Args:
            images_num_small: lower bound
            images_num_large: upper bound
            categories_num: number of categories
        """
        rev = []
        for i in range(len(self.sub_numbers)):
            if self.sub_numbers[i] <= images_num_large and self.sub_numbers[i] >= images_num_small:
                rev.append(self.sub_categories[i])
        random.shuffle(rev,random.random)
        if categories_num != 0:
            rev = rev[:categories_num]
        print("{:d} categories have been selected.".format(len(rev)))
        return rev
    
    def rangeCategoriesSimilar(self, images_num_small, images_num_large, categories_num):
        """
        For giving range of image numbers, and number of categories wanted, pick a list of category names with high occurance words and return
        Args:
            images_num_small: lower bound
            images_num_large: upper bound
            categories_num: number of categories
        """
        rev = []
        for i in range(len(self.sub_numbers)):
            if self.sub_numbers[i] <= images_num_large and self.sub_numbers[i] >= images_num_small:
                rev.append(self.sub_categories[i])
        rev = self.selectSimilar(rev, categories_num)
        print("{:d} categories have been selected.".format(len(rev)))
        return rev
    
    def rangeCategoriesNonSimilar(self, images_num_small, images_num_large, categories_num):
        """
        For giving range of image numbers, and number of categories wanted, pick a list of category names with low occurance words and return
        Args:
            images_num_small: lower bound
            images_num_large: upper bound
            categories_num: number of categories
        """
        rev = []
        for i in range(len(self.sub_numbers)):
            if self.sub_numbers[i] <= images_num_large and self.sub_numbers[i] >= images_num_small:
                rev.append(self.sub_categories[i])
        rev = self.selectNonSimilar(rev, categories_num)
        print("{:d} categories have been selected.".format(len(rev)))
        return rev
    
    def rangeCategoriesSimilarTo(self, images_num_small, images_num_large, categories_num, categories_in):
        """
        For giving range of image numbers, number of categories wanted and source category list, 
        pick a list of category names similar to source category and return
        Args:
            images_num_small: lower bound
            images_num_large: upper bound
            categories_num: number of categories
            categories_in: source category list
        """
        rev = []
        for i in range(len(self.sub_numbers)):
            if self.sub_numbers[i] <= images_num_large and self.sub_numbers[i] >= images_num_small:
                rev.append(self.sub_categories[i])
        rev = self.selectSimilarTo(categories_in, rev, categories_num)
        print("{:d} categories have been selected.".format(len(rev)))
        return rev
    
    def rangeCategoriesNonSimilarTo(self, images_num_small, images_num_large, categories_num, categories_in):
        """
        For giving range of image numbers, number of categories wanted and source category list, 
        pick a list of category names not similar to source category and return
        Args:
            images_num_small: lower bound
            images_num_large: upper bound
            categories_num: number of categories
            categories_in: source category list
        """
        rev = []
        for i in range(len(self.sub_numbers)):
            if self.sub_numbers[i] <= images_num_large and self.sub_numbers[i] >= images_num_small:
                rev.append(self.sub_categories[i])
        rev = self.selectNonSimilarTo(categories_in, rev, categories_num)
        print("{:d} categories have been selected.".format(len(rev)))
        return rev
    
    def selectSimilar(self, categories, num):
        """
        Ranking the categories by the sum of their name word occurance in descending order and return top n results
        Args:
            categories: list of category names
            num: n
        """
        str_List = {}
        cat_List = {}
        for cat in categories:
            strs = cat.split(' ')
            for s in strs:
                if s in str_List:
                    str_List[s] += 1
                else:
                    str_List[s] = 1
        for cat in categories:
            score = 0
            strs = cat.split(' ')
            unique_strs = []
            for x in strs:
                if x not in unique_strs:
                    unique_strs.append(x)
            for s in unique_strs:
                score += str_List[s]
            cat_List[cat] = score
        all_sorted_keys = sorted(cat_List, key=cat_List.get, reverse = True)               
        if num != 0:
            all_sorted_keys = all_sorted_keys[:num]
        return all_sorted_keys
    
    def selectSimilarTo(self, categories_source, categories_target, num):
        """
        Ranking the target category list by the sum of their name word occurance compared with source list in descending order and return top n results
        Args:
            categories_source: list of category names
            categories_target: list of category names for ranking
            num: n
        """
        str_List = {}
        cat_List = {}
        for cat in categories_source:
            strs = cat.split(' ')
            for s in strs:
                if s in str_List:
                    str_List[s] += 1
                else:
                    str_List[s] = 1
        for cat in categories_target:
            score = 0
            strs = cat.split(' ')
            unique_strs = []
            for x in strs:
                if x not in unique_strs:
                    unique_strs.append(x)
            for s in unique_strs:
                if s in str_List:
                    score += str_List[s]
            cat_List[cat] = score
        all_sorted_keys = sorted(cat_List, key=cat_List.get, reverse = True)
        if num != 0:
            all_sorted_keys = all_sorted_keys[:num]
        return all_sorted_keys
    
    def selectNonSimilar(self, categories, num):
        """
        Ranking the categories by the sum of their name word occurance in ascending order and return top n results
        Args:
            categories: list of category names
            num: n
        """
        str_List = {}
        cat_List = {}
        for cat in categories:
            strs = cat.split(' ')
            for s in strs:
                if s in str_List:
                    str_List[s] += 1
                else:
                    str_List[s] = 1
        for cat in categories:
            score = 0
            strs = cat.split(' ')
            unique_strs = []
            for x in strs:
                if x not in unique_strs:
                    unique_strs.append(x)
            for s in unique_strs:
                score += str_List[s]
            cat_List[cat] = score
        all_sorted_keys = sorted(cat_List, key=cat_List.get)                 
        if num != 0:
            all_sorted_keys = all_sorted_keys[:num]
        return all_sorted_keys
    
    def selectNonSimilarTo(self, categories_source, categories_target, num):
        """
        Ranking the target category list by the sum of their name word occurance compared with source list in ascending order and return top n results
        Args:
            categories_source: list of category names
            categories_target: list of category names for ranking
            num: n
        """
        str_List = {}
        cat_List = {}
        for cat in categories_source:
            strs = cat.split(' ')
            for s in strs:
                if s in str_List:
                    str_List[s] += 1
                else:
                    str_List[s] = 1
        for cat in categories_target:
            score = 0
            strs = cat.split(' ')
            unique_strs = []
            for x in strs:
                if x not in unique_strs:
                    unique_strs.append(x)
            for s in unique_strs:
                if s in str_List:
                    score += str_List[s]
            cat_List[cat] = score
        all_sorted_keys = sorted(cat_List, key=cat_List.get)
        if num != 0:
            all_sorted_keys = all_sorted_keys[:num]
        return all_sorted_keys


