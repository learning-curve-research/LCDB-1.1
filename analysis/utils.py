import numpy as np
import scipy
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pathlib import Path
import h5py
from meta_feature import anchor_list_denser, learner_zoo, learner_zoo_abbreviation

# from scipy.stats import t

def get_datasets_and_learner_zoo_for_excluded_learners(excluded_learners):

    # check that excluded learners are contained in the zoo
    for el in excluded_learners:
        if el not in learner_zoo:
            raise ValueError(f"invalid learner name {el}")

    # load datasets only of the included learners
    file_paths = Path.cwd() / '../dataset/LCDB11_ER_CC18_24.hdf5'
    dataset = h5py.File(file_paths , 'r')['error rate'][:, [i for i, n in enumerate(learner_zoo) if n not in excluded_learners]]
    # dataset_nofs, dataset_minmaxfs, dataset_standardfs 
    dataset_nofs, dataset_minmaxfs, dataset_standardfs = [dataset[..., 0, 0], dataset[..., 1, 0], dataset[..., 2, 0]]
    # reduce the learner zoo to the included ones
    _learner_zoo = [(l_long, l_short) for l_long, l_short in zip(learner_zoo, learner_zoo_abbreviation) if l_long not in excluded_learners]
    return [l[0] for l in _learner_zoo], [l[1] for l in _learner_zoo], dataset_nofs, dataset_minmaxfs, dataset_standardfs


def paired_greater_ttest_pvalue(values_greater, values_smaller):
    """
    one side p value: values_greater > values_smaller
    """

    diff = values_greater - values_smaller  # maintain the related 

    n = len(diff)  
    if n < 2: 
        return np.nan
    
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)  
    se_diff = std_diff / np.sqrt(n)  
    
    if se_diff == 0:  # in case exactly the same
        return 1.0

    t_statistic = mean_diff / se_diff

    # one side (alternative='greater')
    p_value = 1 - stats.t.cdf(t_statistic, df=n-1)
    return p_value


def flat_detector(input_dataset, threshold = 0.05):
    # dataset in (72, 24, 5, 5, 137, 3)
    # error array in (24, 72) 
    output_shape = (input_dataset.shape[1], input_dataset.shape[0]) 
    error_matrix_y = np.full(output_shape, np.nan)  
    error_matrix_x = np.full(output_shape, np.nan)
    
    mean_data = np.nanmean(input_dataset, axis=(2, 3))
    mean_data = np.transpose(mean_data[:, :, :, :], (1, 0, 2, 3))
    mean_data_valid = mean_data[:,:,:,1]

    # loop detect straight line
    for data_idx in range(mean_data_valid.shape[1]): 
        all_dataset_max = np.nanmax(mean_data_valid[:, data_idx, :])  
        all_dataset_min = np.nanmin(mean_data_valid[:, data_idx, :])  
        all_dataset_diff = all_dataset_max - all_dataset_min

        for learner_idx in range(mean_data_valid.shape[0]):  
            # one learner curve
            curve = mean_data_valid[learner_idx, data_idx, :]
            
            diff = np.nanmax(curve) - np.nanmin(curve)
            
            # classify (approximate) straght line
            if diff <= (threshold * all_dataset_diff):
                # set zero
                error_matrix_y[learner_idx, data_idx] = -1
                error_matrix_x[learner_idx, data_idx] = -1
    return error_matrix_y, error_matrix_x


def global_monotonicity_violation(input_dataset, flat_filter = False, bonferroni = True, dipping = False, anchor_list = anchor_list_denser): 
    # missing learning curves:        NaN
    # flat learning curves:           -1
    # no significant violation:       0
    # significant violation error:    0-1
    '''
    input_dataset:      the array of learning curves
    flat_filter:        filter the flat curve or not
    dipping:            True: dipping check, False: global mono violation check
    bonferroni:         True/False, alpha corrected by bonferroni method
    '''
    
    # input dataset in (72/265, 24, 5, 5, 137, 3)    
    num_learner = input_dataset.shape[1]
    num_anchor = input_dataset.shape[4]
    group_lc = input_dataset.reshape(-1,num_learner,25,num_anchor,3)
    group_lc_valid = group_lc[:,:,:,:,1]
    group_lc_valid = np.transpose(group_lc_valid, (1, 0, 2, 3)) # (24,72/265,25,137)

    # output matrix in #(24,72/265)
    output_shape = (input_dataset.shape[1], input_dataset.shape[0]) 
    error_matrix_y = np.full(output_shape, np.nan)  
    error_matrix_x = np.full(output_shape, np.nan)

    # loop
    for dataset_idx in tqdm(range(group_lc_valid.shape[1])):
        for learner_idx in range(group_lc_valid.shape[0]): 
            if flat_filter:     # check flatness
                mean_onedataset_valid = np.nanmean(group_lc_valid[:, dataset_idx, :, :], axis = 1)
                minmax_diff = np.nanmax(mean_onedataset_valid) - np.nanmin(mean_onedataset_valid)
                learner_minmax_diff = np.nanmax(mean_onedataset_valid[learner_idx, :]) - np.nanmin(mean_onedataset_valid[learner_idx, :])

                if np.isnan(learner_minmax_diff):
                    continue
                elif learner_minmax_diff > (minmax_diff * 0.05):
                    pass
                else: 
                    error_matrix_y[learner_idx, dataset_idx] = -1
                    error_matrix_x[learner_idx, dataset_idx] = -1
                    continue

            # else mean curve
            curves_group = group_lc_valid[learner_idx, dataset_idx, :, :]
            mean_curve = np.nanmean(curves_group, axis=0)
            if np.all(np.isnan(mean_curve)):
                continue

            # remove nan in curve & align with anchor index
            mask_indices = ~np.isnan(mean_curve)
            curves_group_clean = curves_group[:, mask_indices]
            mean_curve_clean = mean_curve[mask_indices]   
            anchor_list_clean = anchor_list[mask_indices]

            if len(mean_curve_clean) < 2:
                error_matrix_y[learner_idx, dataset_idx] = 0
                error_matrix_x[learner_idx, dataset_idx] = 0
                continue

            # find all possible i, j pairs
            num_points = len(mean_curve_clean)
            pair_indices = [(i, j) for i in range(num_points - 1) for j in range(i + 1, num_points)]
            if dipping: 
                j = num_points -1
                pair_indices = [(i, j) for i in range(num_points - 1)]
            # find i,j
            max_difference = 0 
            i_index, j_index = None, None

            # loop all (i, j) pairs to find the one with the smallest difference
            for (i, j) in pair_indices:
                diff = mean_curve_clean[j] - mean_curve_clean[i]
                if diff > max_difference:
                    max_difference = diff
                    i_index, j_index = i, j
            
            if max_difference == 0 or i_index is None or j_index is None:
                error_matrix_y[learner_idx, dataset_idx] = 0
                error_matrix_x[learner_idx, dataset_idx] = 0
                continue

            # extract the group values at points i and j
            group_values_i = curves_group_clean[:, i_index]
            group_values_j = curves_group_clean[:, j_index]
            
            # remove NaN from both arrays simultaneously to ensure equal length
            valid_mask = ~np.isnan(group_values_i) & ~np.isnan(group_values_j)
            group_values_i = group_values_i[valid_mask]
            group_values_j = group_values_j[valid_mask]

            if len(group_values_i) < 2 or len(group_values_j) < 2:
                error_matrix_y[learner_idx, dataset_idx] = 0
                error_matrix_x[learner_idx, dataset_idx] = 0
                continue
            
            #  paired t-test (one-sided, alternative hypothesis: mean of i > mean of j)
            t_statistic, p_value = stats.ttest_rel(group_values_j, group_values_i, alternative='greater')

            if bonferroni: 
                p_value = p_value * len(pair_indices)
            else: 
                pass

            if p_value >= 0.05: 
                error_matrix_y[learner_idx, dataset_idx] = 0
                error_matrix_x[learner_idx, dataset_idx] = 0
            else:       
                # Y and X
                monotonicity_error_Y = (max_difference) 

                # use ∆ | data | to mitigate occasional fluctuations
                num_data_diff = anchor_list_clean[j_index] - anchor_list_clean[i_index]
                all_num_data = anchor_list_clean[-1] - anchor_list_clean[0]
                non_mono_length = num_data_diff / all_num_data

                error_matrix_y[learner_idx, dataset_idx] = monotonicity_error_Y
                error_matrix_x[learner_idx, dataset_idx] = non_mono_length
            
    return error_matrix_y, error_matrix_x



def global_convexity_violation(input_dataset, flat_filter = False, bonferroni = True, anchor_list = anchor_list_denser): 
    
    # input dataset in (72/265, 24, 5, 5, 137, 3)   
    num_learner = input_dataset.shape[1] 
    num_anchor = input_dataset.shape[4]
    group_lc = input_dataset.reshape(-1,num_learner,25,num_anchor,3)
    group_lc_valid = group_lc[:,:,:,:,1]
    group_lc_valid = np.transpose(group_lc_valid, (1, 0, 2, 3)) # (24,72/265,25,137)

    # output matrix in #(24,72/265)
    output_shape = (input_dataset.shape[1], input_dataset.shape[0]) 
    error_matrix, index_h_matrix, index_i_matrix, index_j_matrix = (np.full(output_shape, np.nan) for _ in range(4))


    # loop
    for dataset_idx in tqdm(range(group_lc_valid.shape[1])):
        for learner_idx in range(group_lc_valid.shape[0]): 
            if flat_filter:     # check flatness
                mean_onedataset_valid = np.nanmean(group_lc_valid[:, dataset_idx, :, :], axis = 1)
                minmax_diff = np.nanmax(mean_onedataset_valid) - np.nanmin(mean_onedataset_valid)
                learner_minmax_diff = np.nanmax(mean_onedataset_valid[learner_idx, :]) - np.nanmin(mean_onedataset_valid[learner_idx, :])

                if np.isnan(learner_minmax_diff):
                    continue
                elif learner_minmax_diff > (minmax_diff * 0.05):
                    pass
                else: 
                    error_matrix[learner_idx, dataset_idx] = -1
                    index_h_matrix[learner_idx, dataset_idx] = None
                    index_i_matrix[learner_idx, dataset_idx] = None
                    index_j_matrix[learner_idx, dataset_idx] = None
                    continue

            # else mean curve
            curves_group = group_lc_valid[learner_idx, dataset_idx, :, :]
            mean_curve = np.nanmean(curves_group, axis=0)
            if np.all(np.isnan(mean_curve)):
                continue
            
            # remove nan in curve & align with anchor index
            mask_indices = ~np.isnan(mean_curve)
            curves_group_clean = curves_group[:, mask_indices]
            mean_curve_clean = mean_curve[mask_indices]   
            anchor_list_clean = anchor_list[mask_indices]

            if len(mean_curve_clean) < 3:
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
                continue

            # all possible h < i < j triples
            num_points = len(mean_curve_clean)
            triple_indices = [(h, i, j) for h in range(num_points - 2) 
                              for i in range(h + 1, num_points - 1) 
                              for j in range(i + 1, num_points)]

            # find maximum convexity violation i
            max_violation = 0 
            h_index, i_index, j_index = None, None, None

            # loop all (h, i, j)
            for (h, i, j) in triple_indices:
                # mid point
                uneven_ratio_j = (anchor_list_clean[i] - anchor_list_clean[h]) / (anchor_list_clean[j] - anchor_list_clean[h]) 
                uneven_ratio_h = (anchor_list_clean[j] - anchor_list_clean[i]) / (anchor_list_clean[j] - anchor_list_clean[h])
                mid_point_hj = mean_curve_clean[h] * uneven_ratio_h + mean_curve_clean[j] * uneven_ratio_j 
                # diff
                violation = mean_curve_clean[i] - mid_point_hj

                if violation > max_violation:
                    max_violation = violation
                    h_index, i_index, j_index = h, i, j
                
            if max_violation == 0 or h_index is None or i_index is None or j_index is None:
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
                continue
            
            # extract the group values at points i and j
            group_values_h = curves_group_clean[:, h_index]
            group_values_i = curves_group_clean[:, i_index]
            group_values_j = curves_group_clean[:, j_index]
            
            # remove NaN from both arrays simultaneously to ensure equal length
            valid_mask = ~np.isnan(group_values_h) & ~np.isnan(group_values_i) & ~np.isnan(group_values_j)
            group_values_h = group_values_h[valid_mask]
            group_values_i = group_values_i[valid_mask]
            group_values_j = group_values_j[valid_mask]

            if np.count_nonzero(valid_mask) < 2:
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
                continue

                # mid_point_hj = mean_curve_clean[h] * uneven_ratio_h + mean_curve_clean[j] * uneven_ratio_j 
                # # diff
                # violation = mean_curve_clean[i] - mid_point_hj
            # mid point
            group_uneven_ratio_j = (anchor_list_clean[i_index] - anchor_list_clean[h_index]) / (anchor_list_clean[j_index] - anchor_list_clean[h_index]) 
            group_uneven_ratio_h = (anchor_list_clean[j_index] - anchor_list_clean[i_index]) / (anchor_list_clean[j_index] - anchor_list_clean[h_index]) 
            group_mid_point_hj = group_values_h * group_uneven_ratio_h + group_values_j * group_uneven_ratio_j
            # diff
            group_violation = group_values_i - group_mid_point_hj

            t_statistic, p_value = stats.ttest_1samp(group_violation, popmean=0, alternative='greater')
            if bonferroni: 
                p_value = p_value * len(triple_indices)
            else: 
                pass

            if p_value >= 0.05: 
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
            else: 
                # normalization
                # max_value = np.max(mean_curve_clean)
                # min_value = np.min(mean_curve_clean)
                error_matrix[learner_idx, dataset_idx] = max_violation # / (max_value - min_value) 
                index_h_matrix[learner_idx, dataset_idx] = anchor_list_clean[h_index]
                index_i_matrix[learner_idx, dataset_idx] = anchor_list_clean[i_index]
                index_j_matrix[learner_idx, dataset_idx] = anchor_list_clean[j_index]

    return error_matrix, index_h_matrix, index_i_matrix, index_j_matrix


def peaking_detection(input_dataset, flat_filter = False, bonferroni = True, anchor_list = anchor_list_denser): 
   
    # input dataset in (72/265, 24, 5, 5, 137, 3)    
    num_learner = input_dataset.shape[1] 
    num_anchor = input_dataset.shape[4]
    group_lc = input_dataset.reshape(-1,num_learner,25,num_anchor,3)
    group_lc_valid = group_lc[:,:,:,:,1]
    group_lc_valid = np.transpose(group_lc_valid, (1, 0, 2, 3)) # (24,72/265,25,137)

    # output matrix in #(24,72/265)
    output_shape = (input_dataset.shape[1], input_dataset.shape[0]) 
    error_matrix, index_h_matrix, index_i_matrix, index_j_matrix = (np.full(output_shape, np.nan) for _ in range(4))


    # loop
    for dataset_idx in tqdm(range(group_lc_valid.shape[1])):
        for learner_idx in range(group_lc_valid.shape[0]): 
            if flat_filter:     # check flatness
                mean_onedataset_valid = np.nanmean(group_lc_valid[:, dataset_idx, :, :], axis = 1)
                minmax_diff = np.nanmax(mean_onedataset_valid) - np.nanmin(mean_onedataset_valid)
                learner_minmax_diff = np.nanmax(mean_onedataset_valid[learner_idx, :]) - np.nanmin(mean_onedataset_valid[learner_idx, :])

                if np.isnan(learner_minmax_diff):
                    continue
                elif learner_minmax_diff > (minmax_diff * 0.05):
                    pass
                else: 
                    error_matrix[learner_idx, dataset_idx] = -1
                    index_h_matrix[learner_idx, dataset_idx] = None
                    index_i_matrix[learner_idx, dataset_idx] = None
                    index_j_matrix[learner_idx, dataset_idx] = None
                    continue

            # else mean curve
            curves_group = group_lc_valid[learner_idx, dataset_idx, :, :]
            mean_curve = np.nanmean(curves_group, axis=0)
            if np.all(np.isnan(mean_curve)):
                continue
            
            # remove nan in curve & align with anchor index
            mask_indices = ~np.isnan(mean_curve)
            curves_group_clean = curves_group[:, mask_indices]
            mean_curve_clean = mean_curve[mask_indices]   
            anchor_list_clean = anchor_list[mask_indices]

            if len(mean_curve_clean) < 3:
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
                continue

            # all possible h < i < j triples
            num_points = len(mean_curve_clean)
            triple_indices = [(h, i, j) for h in range(num_points - 2) 
                              for i in range(h + 1, num_points - 1) 
                              for j in range(i + 1, num_points)]
            pair_indices = [(h, i) for h in range(num_points - 1) for i in range(h + 1, num_points)]

            # find maximum convexity violation i
            max_violation = 0 
            h_index, i_index, j_index = None, None, None

            # loop all (h, i, j)
            for (h, i, j) in triple_indices:
                # mid point
                uneven_ratio_j = (anchor_list_clean[i] - anchor_list_clean[h]) / (anchor_list_clean[j] - anchor_list_clean[h]) 
                uneven_ratio_h = (anchor_list_clean[j] - anchor_list_clean[i]) / (anchor_list_clean[j] - anchor_list_clean[h])
                mid_point_hj = mean_curve_clean[h] * uneven_ratio_h + mean_curve_clean[j] * uneven_ratio_j 
                # diff
                violation = mean_curve_clean[i] - mid_point_hj

                if violation > max_violation:
                    max_violation = violation
                    h_index, i_index, j_index = h, i, j
                
            if max_violation == 0 or h_index is None or i_index is None or j_index is None:
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
                continue
            
            # extract the group values at points i and j
            group_values_h = curves_group_clean[:, h_index]
            group_values_i = curves_group_clean[:, i_index]
            group_values_j = curves_group_clean[:, j_index]
            
            # remove NaN from both arrays simultaneously to ensure equal length
            valid_mask = ~np.isnan(group_values_h) & ~np.isnan(group_values_i) & ~np.isnan(group_values_j)
            group_values_h = group_values_h[valid_mask]
            group_values_i = group_values_i[valid_mask]
            group_values_j = group_values_j[valid_mask]

            if np.count_nonzero(valid_mask) < 2:
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
                continue

                # mid_point_hj = mean_curve_clean[h] * uneven_ratio_h + mean_curve_clean[j] * uneven_ratio_j 
                # # diff
                # violation = mean_curve_clean[i] - mid_point_hj
            # mid point
            group_uneven_ratio_j = (anchor_list_clean[i_index] - anchor_list_clean[h_index]) / (anchor_list_clean[j_index] - anchor_list_clean[h_index]) 
            group_uneven_ratio_h = (anchor_list_clean[j_index] - anchor_list_clean[i_index]) / (anchor_list_clean[j_index] - anchor_list_clean[h_index]) 
            group_mid_point_hj = group_values_h * group_uneven_ratio_h + group_values_j * group_uneven_ratio_j
            # diff
            group_violation = group_values_i - group_mid_point_hj

            # p_value of convex: h,i,j
            _, p_value_convex = stats.ttest_1samp(group_violation, popmean=0, alternative='greater')
            # p_value of mono: h,i
            _, p_value_mono = stats.ttest_rel(group_values_i, group_values_h, alternative='greater')

            if bonferroni: ############## BONFERRONI should be the sum of all testing
                p_value_convex = p_value_convex * ( len(triple_indices) + len(pair_indices) )
                p_value_mono = p_value_mono * ( len(triple_indices) + len(pair_indices) )
            else: 
                pass

            if p_value_convex >= 0.05 or p_value_mono >= 0.05: 
                error_matrix[learner_idx, dataset_idx] = 0
                index_h_matrix[learner_idx, dataset_idx] = None
                index_i_matrix[learner_idx, dataset_idx] = None
                index_j_matrix[learner_idx, dataset_idx] = None
            else: 
                # normalization
                # max_value = np.max(mean_curve_clean)
                # min_value = np.min(mean_curve_clean)
                error_matrix[learner_idx, dataset_idx] = max_violation # / (max_value - min_value) 
                index_h_matrix[learner_idx, dataset_idx] = anchor_list_clean[h_index]
                index_i_matrix[learner_idx, dataset_idx] = anchor_list_clean[i_index]
                index_j_matrix[learner_idx, dataset_idx] = anchor_list_clean[j_index]

    return error_matrix, index_h_matrix, index_i_matrix, index_j_matrix


def identify_local_mono(learner_idx, dataset_idx, input_dataset, threshold_anchor = 0, anchor_list = anchor_list_denser):    # anchor 0(16), 16(64), 32(256)
    # input dataset in (72/265, 24, 5, 5, 137, 3)    
    num_learner = input_dataset.shape[1] 
    num_anchor = input_dataset.shape[4]
    group_lc = input_dataset.reshape(-1,num_learner,25,num_anchor,3)
    group_lc_valid = group_lc[:,:,:,:,1]
    group_lc_valid = np.transpose(group_lc_valid, (1, 0, 2, 3)) # (24,72/265,25,137)
    curves_group = group_lc_valid[learner_idx, dataset_idx, :, :]

    curves_group = curves_group[:, threshold_anchor:]
    mean_curve = np.nanmean(curves_group, axis=0)

    # remove nan in curve & align with anchor index
    mask_indices = ~np.isnan(mean_curve)
    curves_group_clean = curves_group[:, mask_indices]
    mean_curve_clean = mean_curve[mask_indices]   
    anchor_list_clean = anchor_list[threshold_anchor:][mask_indices]

    number_peaks = 0
    if len(mean_curve_clean) < 2:
        print(f"Not a curve in learner {learner_idx} dataset {dataset_idx}")
        monotonicity_list = []
    else:
        # t-test neighbour anchor
        num_points = len(mean_curve_clean)
        monotonicity_list = []

        for i in range(num_points - 1):
            group_values_i = curves_group_clean[:, i]    
            group_values_j = curves_group_clean[:, i+1]  
            # stats can return nan, 0, and 1 randomly
            if np.array_equal(np.round(group_values_i, decimals=7), np.round(group_values_j, decimals=7)):  
                bonferroni_p_value_improved = bonferroni_p_value_worse = 1.0
            else:
                # improved
                _, p_value_improved = stats.ttest_rel(group_values_i, group_values_j, alternative='greater')
                # worsen
                _, p_value_worsen = stats.ttest_rel(group_values_j, group_values_i, alternative='greater')

                bonferroni_p_value_improved = p_value_improved * (num_points - 1)
                bonferroni_p_value_worse = p_value_worsen * (num_points - 1)

            if bonferroni_p_value_improved < 0.05:
                monotonicity_list.append(1)  # error rate decrease
            elif bonferroni_p_value_worse < 0.05:
                monotonicity_list.append(-1)  # error rate increase
            else:
                monotonicity_list.append(0)
        monotonicity_list.append(0) # the last anchor 0

        # count peaks in the mono_list
        state = 0
        for value in monotonicity_list:
            if value == -1:
                state = -1  # enter state mark

            if value == 1 and state == -1:
                number_peaks += 1  # peak detected when value switching from -1 to 1
                state = 0  # reset state after detecting a peak


    return monotonicity_list, anchor_list_clean, number_peaks



####### fitting

def get_num_par(model_id):
    if model_id == 'last1':
        return 1
    if model_id in ['pow2', 'log2', 'exp2', 'lin2', 'ilog2']:
        return 2
    if model_id in ['pow3', 'exp3', 'vap3', 'expp3', 'expd3', 'logpower3']:
        return 3
    if model_id in ['mmf4', 'wbl4', 'exp4', 'pow4', 'janoschek4']:
        return 4


def fit_model(sizes, scores, sizes_extrapolation, model_id, rep=5, verbose=True):
    sizes = np.array(sizes)
    scores = np.array(scores)

    bad_score = np.isnan(scores)

    sizes = sizes[bad_score == False]
    scores = scores[bad_score == False]

    # this defines the curve model
    def get_fun(beta):
        num_par = get_num_par(model_id)
        fun = None

        # unpack parameters
        if num_par == 1:
            a = beta[0]
        if num_par == 2:
            a, b = beta[0], beta[1]
        if num_par == 3:
            a, b, c = beta[0], beta[1], beta[2]
        if num_par == 4:
            a, b, c, d = beta[0], beta[1], beta[2], beta[3]

        # define curve models
        if model_id == 'pow2':
            fun = lambda x: -a * x ** (-b)
        if model_id == 'pow3':
            fun = lambda x: a - b * x ** (-c)
        if model_id == 'log2':
            fun = lambda x: -a * np.log(x) + b
        if model_id == 'exp3':
            fun = lambda x: a * np.exp(-b * x) + c
        if model_id == 'exp2':
            fun = lambda x: a * np.exp(-b * x)
        if model_id == 'lin2':
            fun = lambda x: a * x + b
        if model_id == 'vap3':
            fun = lambda x: np.exp(a + b / x + c * np.log(x))
        if model_id == 'mmf4':
            fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
        if model_id == 'wbl4':
            fun = lambda x: (c - b * np.exp(-a * (x ** d)))
        if model_id == 'exp4':
            fun = lambda x: c - np.exp(-a * (x ** d) + b)
        if model_id == 'expp3':
            # fun = lambda x: a * np.exp(-b*x) + c
            fun = lambda x: c - np.exp((x - b) ** a)
        if model_id == 'pow4':
            fun = lambda x: a - b * (x + d) ** (-c)  # has to closely match pow3
        if model_id == 'ilog2':
            fun = lambda x: b - (a / np.log(x))
        if model_id == 'expd3':
            fun = lambda x: c - (c - a) * np.exp(-b * x)
        if model_id == 'logpower3':
            fun = lambda x: a / (1 + (x / np.exp(b)) ** c)
        if model_id == 'last1':
            fun = lambda x: (a + x) - x  # casts the prediction to have the correct size
        if model_id == 'janoschek4': 
            fun = lambda x: a - b * np.exp(-c * x ** d)
        return fun

    def objective(beta):  # this returns the residuals of the fit on the training points
        fun = get_fun(beta)
        return fun(sizes) - scores

    # we dp multiple repititions and collect best results in lists below
    beta_list = []
    trn_error = []

    # this model requires no optimization
    if model_id == 'last1':
        a = scores[-1]
        return np.array([a]), get_fun(np.array([a])), 0, 0

    # failure statistics
    #rep = 5
    fails_fit = 0
    fails_init = 0
    i = 0
    init = None

    while i <= rep:  # try repeatedly to fit a model
        num_par = get_num_par(model_id)

        beta = None
        error = True
        first = True
        # keep trying initial points until a suitable one is found
        while (error):

            if fails_init > 100 or fails_fit > 20:  # give up
                best_beta = np.zeros(num_par)
                # if verbose:
                #     print('giving up...')
                return best_beta, get_fun(best_beta), fails_init, fails_fit

            if not first:
                fails_init += 1
                # if verbose:
                    # print('initial value failed, retrying for ', model_id)
            init = np.random.rand(num_par)

            if model_id == 'pow4':  # this init works well for pow4
                best_beta, _, _, _ = fit_model(sizes, scores, sizes_extrapolation, 'pow3')
                init[0:3] = best_beta

            # check for errors in initial point
            trn_error_init = np.mean(objective(init) ** 2)
            fun_init = get_fun(init)
            sizes_all = np.hstack((sizes, sizes_extrapolation))
            hat_all = fun_init(sizes_all)
            nan_error1 = np.isnan(hat_all).any()
            inf_error1 = np.isinf(hat_all).any()
            nan_error2 = np.isnan(trn_error_init).any()
            inf_error2 = np.isinf(trn_error_init).any()
            error = nan_error1 or inf_error1 or nan_error2 or inf_error2

            first = False

        # start fitting
        beta = scipy.optimize.least_squares(objective, init, method="lm").x

        # check if fit extrapolates well to unseen sizes
        fun = get_fun(beta)
        extrapolations = fun(sizes_extrapolation)
        nan_error = np.isnan(extrapolations).any()
        inf_error = np.isinf(extrapolations).any()

        if nan_error or inf_error:
            pass  # redo's the optimization since extrapolations failed
            fails_fit += 1
            if verbose:
                print('fit failed, nan error?', nan_error, 'inf error?', inf_error, 'model?', model_id)
        else:
            i += 1
            pass  # save the parameter values and objective function
            beta_list.append(beta)
            trn_error.append(np.mean(objective(beta) ** 2))

    # select the best one
    trn_error = np.array(trn_error)
    best_i = np.argmin(trn_error)

    best_beta = beta_list[best_i]
    return best_beta, get_fun(best_beta), fails_init, fails_fit

def curves_models_fitting(lc_data, model_names, extrapolate, mask_anchor_number, eval_anchor_number, rep=10, verbose=False):
    fitting_results = []

    # remove nan in curve & align with anchor index
    mask_indices = ~np.isnan(lc_data)
    scores = lc_data[mask_indices]
    schedule = anchor_list_denser[mask_indices]
    
    for model_name in model_names:

        try:    # fitting
            if extrapolate: 
                train_schedule = np.array(schedule)[:-mask_anchor_number] # training length
                regress_target = np.array(scores)[:-mask_anchor_number]
            else: 
                train_schedule = np.array(schedule) 
                regress_target = np.array(scores)
            beta, model, fails_init, fails_fit = fit_model(
                train_schedule, regress_target, 
                np.array(schedule),  # extrapolation length
                model_name, rep=rep, verbose=verbose)
            
            # predictions 
            predictions = model(np.array(schedule))
            ######################### avoid extreme large MSE
            # predictions[predictions>1] = 1
            # predictions[predictions<0] = 0
            #########################
            # MSE between predictions and actual scores
            if extrapolate: 
                mse = mean_squared_error(np.array(scores)[-eval_anchor_number:], predictions[-eval_anchor_number:])
            else: # whole curve
                mse = mean_squared_error(np.array(scores), predictions)

            fitting_results.append({
                "schedule": schedule,
                "scores": scores,
                "predictions": predictions,
                "mse": mse,      
                "curve_model": model_name,
                "beta": beta,
                "fails_init": fails_init,
                "fails_fit": fails_fit
            })

        except Exception as e:
            # failed fitting
            if verbose:
                print(f"Failed to fit model {model_name} for current learning curve. Error: {e}")
        
    return fitting_results

def successive_halving(learning_curves, budget, budget_increase, dropout_rate, active_mask=None, _history={}):
    
    # only on first call, activate all algorithms
    if active_mask is None:
        active_mask = np.ones(len(learning_curves))
    
    # append the current ensemble
    indices_of_active_algorithms = [int(i) for i in np.where(active_mask)[0]]
    _history[budget] = indices_of_active_algorithms
    
    # recursive cancellation
    current_population_size = len(indices_of_active_algorithms)
    if current_population_size <= 1 or budget >= learning_curves.shape[1]:
        return _history
    
    # determine currently best
    new_population_size = max(1, (current_population_size - dropout_rate) if dropout_rate >= 1 else int(np.round(current_population_size * (1 - dropout_rate))))
    sorted_performances = np.argsort(learning_curves[:, budget])
    print(sorted_performances)
    survivors = [int(i) for i in sorted_performances if i in indices_of_active_algorithms][:new_population_size]
    new_active_mask = np.array([i in survivors for i in range(len(learning_curves))])
    
    # recurse
    return successive_halving(
        learning_curves=learning_curves,
        budget=budget + budget_increase,
        budget_increase=budget_increase,
        dropout_rate=dropout_rate,
        active_mask=new_active_mask,
        _history=_history
    )
