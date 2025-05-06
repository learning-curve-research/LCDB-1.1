import h5py
import numpy as np
from tqdm import tqdm

input_files = ['experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_72.hdf5', 
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_12.hdf5',    # latest file first in order
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_11.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_10.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_9.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_8.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_7.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_6.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_5.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_4.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_3.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_2.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_1.3.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_1.2.hdf5', 
               'experiments/ER_265_DBextract/LCDB11_ER_265_nofs_minmax_198_1.1.hdf5', 
               'experiments/ER_265_DBextract/LCDB11_ER_265_standard_72.hdf5',
               'experiments/ER_265_DBextract/LCDB11_ER_265_standard_198_1.hdf5', 
               'experiments/ER_265_DBextract/LCDB11_ER_265_standard_198_2.hdf5', 
               'experiments/ER_265_DBextract/LCDB11_ER_265_standard_198_3.hdf5']

output_file = 'LCDB11_ER_265.hdf5'
merged_data = None 

for file_path in tqdm(input_files):
    with h5py.File(file_path, 'r') as f:
        data = f['error rate'][...]  
    
        if merged_data is None:
            merged_data = np.copy(data)
        else:
            nan_mask = np.isnan(merged_data)  
            print(f"this round contain {np.sum(nan_mask)} NaN value")
            merged_data[nan_mask] = data[nan_mask]  

with h5py.File(output_file, 'w') as f_out:
    f_out.create_dataset('error rate', data=merged_data, compression='gzip', compression_opts=9)