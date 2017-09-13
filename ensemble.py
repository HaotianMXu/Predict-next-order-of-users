# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:25:52 2017

@author: ht
"""
#py3
import pandas as pd

p1 = pd.read_csv('data/ye/sub_op2.csv')
p3 = pd.read_pickle('data/arb_7000.pkl')

pm=p1.merge(p3,how="left",on=['order_id','product_id'])
#pm=pm.merge(p3,how="left",on=['order_id','product_id'])

pm['reordered']=0.5*pm.reordered+0.5*pm.prediction#_x+0.5*pm.prediction_y
pm.drop(['prediction'],axis=1,inplace=True)
pm.to_csv('sub_op2+arb.csv',index=False)