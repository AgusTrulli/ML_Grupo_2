import os
import pandas as pd
import numpy as np
from datetime import date

from src.cv.data.utils import datasets_paths

def get_features_csv():
    df_features = pd.read_csv(datasets_paths['features_csv'], 
                                   sep=',', decimal=',', encoding='utf-8')
    
    return df_features

def get_partitions_csv():
    df_partitions = pd.read_csv(datasets_paths['partitions_csv'], 
                             sep=',', decimal=',', encoding='utf-8')
    
    return df_partitions

def get_images_path():
    path_images = datasets_paths['images']
    
    return path_images