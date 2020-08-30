# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:33:14 2020

@author: Matthew Wilson
"""
import os, argparse, json
import numpy as np

# with open("params.json") as paramfile:
#     params = json.load(paramfile)
#     print(params['model_setup']['n_galaxy'])


x = np.arange(0,1e7)
for i in x:
    y = np.copy(x)