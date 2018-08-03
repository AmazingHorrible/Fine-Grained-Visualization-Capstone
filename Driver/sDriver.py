# -*- coding: utf-8 -*-
"""
Created on Thu May 10 18:16:51 2018

@author: hangz
"""

from statistic import staticAnalysis
import numpy as np

analyzer = staticAnalysis('Mammalia')

analyzer.statisticSummary()
print('--------------------------------------------------------------')
#analyzer.printNum()
analyzer.printSum()
#print( analyzer.greaterThanCategories(200,0) )
#print( analyzer.smallerThanCategories(50,0) )

#print(analyzer.rangeCategories(0, 100, 0))
#print(analyzer.rangeCategories(100, 200, 0))
print(analyzer.greaterThanCategories(1000, 0))