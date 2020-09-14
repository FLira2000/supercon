# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:51:43 2020

@author: Josias da Silva Junior & Fabio Lira
"""

print('Please wait.')

import pandas as pd
import numpy as np

#loading data
AllElements = pd.read_csv('allElements_unique_m.csv')
                     
def Correlations():
    CorrelationList = []
    FeaturesList = AllElements.columns
    FeaturesList = FeaturesList.drop('Unnamed: 0').drop('material')
    FeaturesList
    for Features in FeaturesList:
        if Features == 'critical_temp':
            continue
        CorrelationList.append([Features, 
                                AllElements['critical_temp'].corr(AllElements[Features],
                                                                  method='spearman')])
        CorrelationList
        GoodCorrelationList = []
        for i in CorrelationList:
            if i[1] > 0.85 or i[1] < -0.85:
                GoodCorrelationList.append( i )   
        
    if len(GoodCorrelationList) > 0:
        return GoodCorrelationList
    else:
        return FeaturesList.drop('critical_temp').tolist()
