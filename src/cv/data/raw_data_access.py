import os
import pandas as pd
import numpy as np
from datetime import date

from src.cv.data.utils import datasets_paths

def get_features_csv():
    # Read credit record CSV file provided.
    df_features = pd.read_csv(datasets_paths['features_csv'], 
                                   sep=',', decimal=',', encoding='utf-8')
    
    return df_features

def get_partitions_csv():
    # Read application record CSV file provided.
    df_partitions = pd.read_csv(datasets_paths['partitions_csv'], 
                             sep=',', decimal=',', encoding='utf-8')
    
    return df_partitions
