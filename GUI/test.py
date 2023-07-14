# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:50:52 2023

@author: Yuanfei Cheng
"""

import socket
import time
from tkinter import * 
from tkinter import ttk,filedialog,messagebox
from numpy import mean,std
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import fun
import torch
import torch.nn as nn
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams [ 'axes.unicode_minus' ]= False

patha1=r"E:\lab\αβ例子GUI\GUI代码\test\基线.pkl"
with open(patha1, 'rb') as file:
     mu_all = pickle.load(file)
plt.figure(figsize=(10,4),dpi=150)
plt.title("基线分布统计")
plt.hist(mu_all, bins = 15,rwidth=0.9,align='left')
plt.show()