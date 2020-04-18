import os
import pandas as pd
from datetime import date

from utils import datasets_paths

def get_credit_record():
    # Read credit record CSV file provided.
    df_credit_record = pd.read_csv(datasets_paths['credit_record'], sep=';', decimal=',')
    
    return df_credit_record

def get_application_record():
    # Read application record CSV file provided.
    df_application_record = pd.read_csv(datasets_paths['application_record'], sep=';', decimal=',')
    
    return datasets_paths['application_record']
