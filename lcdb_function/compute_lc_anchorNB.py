import sys
import json
import random
import time
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from pynisher import limit, MemoryLimitException, WallTimeoutException
# will enter lcdb_function path through apptainer
from mixNB import MixedNBClassifier, get_entry_mixNB
from lcdb import get_dataset_naive_bayes
from directencoder import DirectEncoder

import logging



def process_jobs(dataset, jobs_todo, show_progress=True, error="raise", verbose=False, encoder=DirectEncoder(), results_file='test.json', status_file='status.csv'):

    old_preprocessor = None
    
    X = None
    y = None

    for (i, job) in enumerate(jobs_todo):
        if show_progress:
            print(f'Current Progress on this dataset: %d of %d...' % (i,len(jobs_todo)))

        preprocessor = {'feature_scaling': job['feature_scaling'], 'mix': job['mix'], 'realistic': job['realistic']}
        reload_data = not (preprocessor == old_preprocessor)

        if reload_data:
            print('Loading the dataset...')
            if job['realistic'] == True: 
                X, y, categorical_colunm = get_dataset_naive_bayes(dataset, feature_scaling=False, preprocess=False)
            else: 
                X, y, categorical_colunm = get_dataset_naive_bayes(dataset, feature_scaling=job['feature_scaling'], preprocess=True)
            labels = list(np.unique(y))
            is_binary = len(labels) == 2
            minority_class = labels[np.argmin([np.count_nonzero(y == label) for label in labels])]
            print(f"Labels are: {labels}")
            print(f"minority_class is {minority_class}")
            print("Now computing the anchor ")
            old_preprocessor = preprocessor.copy()

        algorithm = job["learner"]

        # seeding should be done just before computing the learning curve
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)

        # Mapping of algorithms to their corresponding parameters
        defined_params = {
            "mixBernoulliNB": ("MixedNBClassifier", {"categorical_mask": categorical_colunm, "unfitted_nb_numerical": BernoulliNB()}),
            "mixGaussianNB": ("MixedNBClassifier", {"categorical_mask": categorical_colunm, "unfitted_nb_numerical": GaussianNB()}),
            "mixMultinomialNB": ("MixedNBClassifier", {"categorical_mask": categorical_colunm, "unfitted_nb_numerical": MultinomialNB()}),
            "mixComplementNB": ("MixedNBClassifier", {"categorical_mask": categorical_colunm, "unfitted_nb_numerical": ComplementNB()}),
        }

        # Get algorithm and parameters
        if algorithm in defined_params:
            learner_name, learner_params = defined_params[algorithm]
        else:
            print("Not a mix NB.")

        outer_seed = job["outer_seed"]
        inner_seed = job["inner_seed"]
        anchor = job["size_train"]
        timelimit = job["timelimit"]
        realistic = job["realistic"]
        feature_scaling = job["feature_scaling"]

        print(f"learner={algorithm}, size_train={anchor}, outer_seed={outer_seed}, inner_seed={inner_seed}")

        info = None
        status = "init"

        get_entry_mixNB_pynisher = limit(get_entry_mixNB, wall_time=(timelimit, "h"), context='spawn')
        # the default context sometimes runs into errors on the cluster
        # the error is: `[Errno 24] Too many open files`
        # actually, this error may be caused by something else... its resolved via 'ulimit -n 8000'

        try:
            info = get_entry_mixNB_pynisher(learner_name, learner_params, X, y, anchor, outer_seed, inner_seed, realistic, feature_scaling, encoder=encoder, verbose=verbose)
            # pred_acc = accuracy_score(info['y_test'] , info['y_hat_test'])
            # print("test accuracy: ", pred_acc)
            status = "ok"
            for key in job.keys():
                if key in info:
                    pass
                else:
                    info[key] = job[key]
            

        except WallTimeoutException: 
            error_message = f"Time out for %.2f hours one anchor. " % timelimit
            print(error_message)

            info = {
                    "error_type": 'timeout',
                    "error_message": error_message,
                }
            for key in job.keys():
                if key in info:
                    pass
                else:
                    info[key] = job[key]
            status = "timeout"
        
        except Exception as e:
            print(f"AN ERROR OCCURED! Here are the details.\n{type(e)}\n{e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print(traceback_string)

            info = {
                    "error_type": str(type(e)),
                    "error_message": str(e),
                    "error_traceback": traceback_string,
                }
            for key in job.keys():
                if key in info:
                    pass
                else:
                    info[key] = job[key]
            status = "error"

        info["status"] = status
    
        json.dump(info, outfile)
        outfile.write('\n')  # Write a newline character after the first dump
        outfile.flush() # forces to write to the file to make sure data is not lost if we are killed

        statusfile.write('%d,%s\n' % (job["jobid"],status))
        statusfile.flush()


                
if __name__ == '__main__':

    # logger.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)  # Set this to the same as logger level or adjust as needed
    # logger.addHandler(handler)

    job_id = int(sys.argv[1]) # job id integer
    
    print(f'working on job {job_id}...\n')

    # Define base directories
    experiments_dir = os.path.join("..", "jobs")
    results_dir = os.path.join("..", "results")

    # Ensure the directories exist
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Construct file paths using os.path.join
    fn_todo = os.path.join(experiments_dir, f"experiments_job{job_id}.csv")
    fn_results = os.path.join(results_dir, f"result_{job_id}.json")
    fn_status = os.path.join(results_dir, f"status_{job_id}.csv")

    # Print the file paths for verification
    print(f"Todo file path: {fn_todo}")
    print(f"Results file path: {fn_results}")
    print(f"Status file path: {fn_status}")

    my_jobs = pd.read_csv(fn_todo)

    openml_ids = my_jobs['openmlid'].unique()

    print('I have %d jobs! ' % len(my_jobs))
    print('in my job there are %d datasets...\n' % len(openml_ids))
    print(openml_ids)

    # compute jobs
    if not os.path.exists(os.path.dirname(fn_status)):
        os.makedirs(fn_status)
        
    if not os.path.exists(os.path.dirname(fn_results)):
        os.makedirs(fn_results)
    
    with open(fn_status, 'w') as statusfile:
        with open(fn_results, 'w') as outfile:  # "w" is overwrite
            statusfile.write('jobid,status\n')

            start_time = time.time()

            for my_id in openml_ids:

                print('working on openmlid %d...\n' % my_id)
                selected_rows = my_jobs[my_jobs['openmlid'] == my_id]

                selected_rows = selected_rows.sort_values(by=['feature_scaling', 'mix', 'realistic'])
                
                if len(selected_rows['openmlid'].unique()) == 1:
                    dataset = int(selected_rows['openmlid'].unique()[0])
                    print('one dataset, ok')
                else:
                    raise("more than one dataset, failing....")

                jobs_list = []

                for index, row in selected_rows.iterrows():
                    row_dict = row.to_dict()
                    jobs_list.append(row_dict)    
                
                encoder = DirectEncoder(2)
            
                process_jobs(dataset, 
                            jobs_todo = jobs_list, 
                            error="message", 
                            encoder=encoder, 
                            verbose=False, 
                            show_progress=True, 
                            results_file=outfile, 
                            status_file=statusfile)
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('I finished %d jobs! ' % len(my_jobs))
            print(f"Time taken: {elapsed_time} seconds")
