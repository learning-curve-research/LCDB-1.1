import pandas as pd
from lcdb import compute_full_curve, get_dataset, get_entry
import sys
import json
from directencoder import DirectEncoder
import traceback
from tqdm import tqdm
import numpy as np
import sys
import logging
import openml

logger = logging.getLogger('lcdb')
                
if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Set this to the same as logger level or adjust as needed
    logger.addHandler(handler)

    my_jobs = pd.read_csv('missing_experiments_lcdb1.csv')

    openml_ids = my_jobs['openmlid'].unique()

    for index, my_id in enumerate(openml_ids):
        print('working on dataset %d of %d (dataset %d)...\n ' % (index, len(openml_ids), int(my_id)))
        
        ds = openml.datasets.get_dataset(int(my_id))
        logger.info(f"Reading in full dataset from openml API.")
        df = ds.get_data()[0]
        num_rows = len(df)
        logger.info(f"Finished to read original data frame. Size is {len(df)} x {len(df.columns)}.")     

    



