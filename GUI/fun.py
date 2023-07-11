# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:19:58 2023

@author: Yuanfei Cheng
"""

def first_order_filter(data, alpha):
    filtered_data = [data[0]]
    for i in range(1, len(data)):
        filtered_data.append(alpha * data[i] + (1 - alpha) * filtered_data[-1])
    return filtered_data