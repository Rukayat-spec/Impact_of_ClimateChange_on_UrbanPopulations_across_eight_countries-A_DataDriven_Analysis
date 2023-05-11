# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:18:50 2023

@author: ibrah
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cluster_tools as ct


co2_df = pd.read_csv('GDP_per_capital.csv', skiprows= 4)
co2_data = co2_df[["1974", "1984", "1994", "2004", "2014"]]
co2_data.describe()
co2_data.shape

co3_data= co2_data.dropna(axis=0)

ct.map_corr(co3_data)
plt.show()
