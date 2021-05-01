from submission_model_task_4_eval import Submission
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os
import argparse
from copy import deepcopy
import datetime
import warnings
warnings.filterwarnings("ignore")

# as per the metadata file, input and output directories are the arguments
parser = argparse.ArgumentParser(description='Parse input and output paths')
parser.add_argument('--input_dir', default='data/test_data/', type=str, help='directory of the ground-truth evaluation data')
parser.add_argument('--output_dir', default='results/', type=str, help='the directory to save the evaluation output')
parser.add_argument('--ref_data', default='test_private_task_4_more_splits.csv', type=str, help='the ground-truth evaluation data file name')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
reference_file_name = args.ref_data

# make the output dir if not exist
if not os.path.isdir(output_dir):
    os.mkdir(output_dir) 

def pivot_df(df, values):
    """
    Convert dataframe of question and answerrecords to pivoted array, filling in missing columns if some questions are 
    unobserved.
    """ 
    data = df.pivot(index='UserId', columns='QuestionId', values=values)

    # Add rows for any questions not in the test set
    data_cols = data.columns
    all_cols = np.arange(948)
    missing = set(all_cols) - set(data_cols)
    for i in missing:
        data[i] = np.nan
    data = data.reindex(sorted(data.columns), axis=1)

    data = data.to_numpy()
    data[np.isnan(data)] = -1
    return data

if __name__ == "__main__":
    SAVE_NAME = 'real_rand_reinforce02_1'
    METHOD = 'rand'
    data_path = os.path.join(input_dir, reference_file_name)
    df = pd.read_csv(data_path)
    data = pivot_df(df, 'AnswerValue')
    binary_data = pivot_df(df, 'IsCorrect')

    #### EVALUATION
    result_msg= ''
    accs = []

    for SPLIT in range(10):
        print('SPLIT', SPLIT)
        print(datetime.datetime.now())
        ### CREATE MODEL
        M = Submission(SAVE_NAME=SAVE_NAME, METHOD=METHOD)

        # Array containing -1 for unobserved, 0 for observed and not target (can query), 1 for observed and target (held out
        # for evaluation).
        targets = pivot_df(df, 'IsTarget_{}'.format(SPLIT))

        observations = np.zeros_like(deepcopy(data))
        masked_data = deepcopy(data) * observations
        masked_binary_data = deepcopy(binary_data) * observations
        can_query = (deepcopy(data) != -1).astype(int)

        for i in range(10):
            msg = 'Feature selection step {}\n'.format(i+1)
            print(msg)
            result_msg += msg
            next_questions = M.select_feature(masked_data, masked_binary_data, can_query)
            # Validate not choosing previously selected question here

            for i in range(can_query.shape[0]):
                # Validate choosing queriable target here
                assert can_query[i, next_questions[i]] == 1
                can_query[i, next_questions[i]] = 0

                # Validate choosing unselected target here
                assert observations[i, next_questions[i]] == 0

                observations[i, next_questions[i]] = 1
                masked_data = data * observations
                masked_binary_data = binary_data * observations

            # Update model with new data, if required
            M.update_model(masked_data, masked_binary_data, can_query)

        preds = M.predict(masked_data, masked_binary_data)

        pred_list = preds[np.where(targets==1)]
        target_list = binary_data[np.where(targets==1)]
        acc = (pred_list == target_list).astype(int).sum()/len(target_list)
        accs.append(acc)


    #### WRITE RESULTS
    score = np.array(accs).mean()
    std = np.array(accs).std()
    with open(os.path.join(output_dir, 'scores_4_' + SAVE_NAME + '.txt'), 'w') as output_file:
        output_file.write("score:{0}\n".format(score))

    # output detailed results
    with open(os.path.join(output_dir, 'scores_4_' + SAVE_NAME + '.html'), 'w') as output_file:
        htmlString = '''<!DOCTYPE html>
                        <html>
                        <p>phase 4: personalized question</p>
                        </br>
                        <p>overall accuracy: {} +/- {}</p>
                        </br>
                        <p> accuracy for seed 0: {} </p>
                        <p> accuracy for seed 1: {} </p>
                        <p> accuracy for seed 2: {} </p>
                        <p> accuracy for seed 3: {} </p>
                        <p> accuracy for seed 4: {} </p>
                        <p> accuracy for seed 5: {} </p>
                        <p> accuracy for seed 6: {} </p>
                        <p> accuracy for seed 7: {} </p>
                        <p> accuracy for seed 8: {} </p>
                        <p> accuracy for seed 9: {} </p>
        
                        </html>'''.format(score, std, accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9])
        output_file.write(htmlString)