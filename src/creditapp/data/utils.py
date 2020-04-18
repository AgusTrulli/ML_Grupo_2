import os.path as op

import numpy as np
import pandas as pd

PROJECT_PATH = op.abspath(op.join(op.dirname(__file__), '..', '..', '..'))

# Set dataset paths
def data_path(*joins):
    base_path = op.join(PROJECT_PATH, 'data')

    return op.join(base_path, *joins)

APPLICATION_RECORD_DATASET_PATH = data_path('raw', 'application_record.csv')
CREDIT_RECORD_DATASET_PATH = data_path('raw', 'credit_record.csv')

datasets_paths = {
    'application_record': APPLICATION_RECORD_DATASET_PATH,
    'credit_record': CREDIT_RECORD_DATASET_PATH
}