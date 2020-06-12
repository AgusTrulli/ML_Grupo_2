import os.path as op

import numpy as np
import pandas as pd

PROJECT_PATH = op.abspath(op.join(op.dirname(__file__), '..', '..', '..'))

# Set dataset paths
def data_path(*joins):
    base_path = op.join(PROJECT_PATH, 'data')

    return op.join(base_path, *joins)

FEATURES = data_path('raw', 'list_attr_celeba.csv')
PARTITIONS = data_path('raw', 'list_eval_partition.csv')

datasets_paths = {
    'features_csv': FEATURES,
    'partitions_csv': PARTITIONS
}