import pandas as pd 
from tqdm import tqdm
import numpy as np 

file_index = 28
target_split = 1000

# sparse_openml_ids = {273, 293, 351, 354, 357, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 1575, 1592, 4137}  # only for feature scaling
memory_openml_ids = {1111, 1112, 1114, 42732, 42733}

# Load jobs files
dfs = []
num = 0

for i in range(0, 1000):
    try:
        fn = f'/lcdb11_raw/jobs{file_index}/experiments_job{i}.csv'
        df_temp = pd.read_csv(fn)
        dfs.append(df_temp)
        num += len(df_temp)
    except:
        print('file ' + fn + ' is missing...')
        pass

df_jobs = pd.concat(dfs)
print("Total jobs:", len(df_jobs))

# for sparse (not used anymore)
# df_jobs['feature_scaling'] = df_jobs['feature_scaling'].astype(str)
# df_jobs = df_jobs[~ (df_jobs['openmlid'].isin(sparse_openml_ids) & (df_jobs['feature_scaling'] == 'True'))]
# for out of memory
df_jobs = df_jobs[~ (df_jobs['openmlid'].isin(memory_openml_ids))]
print("Total jobs after excluding specific openml IDs:", len(df_jobs))

dfs = []
num = 0
empty_files = []  # To store filenames with only headers

for i in range(0, 1000):
    try:
        fn = f'/lcdb11_raw/results{file_index}/status_{i}.csv'
        df_temp = pd.read_csv(fn)
        # Check if the DataFrame only contains headers (no rows)
        if df_temp.empty:
            empty_files.append(fn)
        else:
            dfs.append(df_temp)
            num += len(df_temp)
    except:
        print('file ' + fn + ' is missing...')
        pass

# Display files that only contain headers (jobid and status)
if empty_files:
    print("Files with only headers and no data:")
    for empty_file in empty_files:
        print(empty_file)
else:
    print("No empty files with only headers found.")

df_status = pd.concat(dfs)

# Filter finished jobs
df_done = df_status[
    (df_status['status'] == 'ok') | 
    (df_status['status'] == 'timeout') | 
    (df_status['status'] == 'error')
]
print("finished jobs:", len(df_done))

done_jobids = set(df_done['jobid'])

# Use the set for checking membership in df_jobs and exclude jobid 
df_filtered_jobs = df_jobs[~df_jobs['jobid'].isin(done_jobids)]

# Split the filtered DataFrame and save
splits = np.array_split(df_filtered_jobs, target_split)

for index, value in enumerate(splits):
    print('working on job %d that consists of %d tasks...\n' % (index, len(value)))
    value.to_csv(f'/lcdb11_raw/jobs/experiments_job{index}.csv', index=False)  # Disable index