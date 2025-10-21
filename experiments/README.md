# Workflow ans Experiments
To ensure the reproducibility, please download the [container](https://surfdrive.surf.nl/files/index.php/s/TSe0nqWKcT5jPwK) or use the [`env`](./env) to generate the same image. 

1. Ensure there are three folder under path `/lcdb1.1`: `jobs`, `logs`, `results`. 
2. Run `jobs_create.py` to create the jobs of generating learning curves with different setting combinations. 
3. Submit the jobs. 
4. Rename the `jobs`, `logs`, `results` folder as `jobs1`, `logs1`, `results1`, ..., and create a new `jobs`, `logs`, `results` folder
5. Check the status of jobs and create jobs for resubmission by `jobs_resubmit.py`. 
6. Repeat step 3-5 until all jobs are done. 
7. Use `jobs_gen_database.py` to generate the dataset. 

Note, both standardization and min-max normalization use `Feature Scaling = True` as the setting. 
After completing all experiments involving one type of scaling (e.g. min-max feature scaling), change the numerical pipeline from `minmaxscaler` to `standardscaler`, and add one line `standardscaler` for categorical pipeline. (line with comment `###### standard scaling case #######`)