from submission_model_task_4 import Submission
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os


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

def validate(selected_users, question_encodings, SAVE_NAME, METHOD):
    # data_path = os.path.normpath('data/test_input/valid_task_4.csv')
    data_path = os.path.normpath('data/test_data/test_public_task_4_more_splits.csv')
    df = pd.read_csv(data_path)
    data = pivot_df(df, 'AnswerValue')[selected_users]
    print('Number of users:', data.shape)
    binary_data = pivot_df(df, 'IsCorrect')[selected_users]

    # Array containing -1 for unobserved, 0 for observed and not target (can query), 1 for observed and target (held out
    # for evaluation).
    targets = pivot_df(df, 'IsTarget_0')[selected_users]

    observations = np.zeros_like(data)
    masked_data = data * observations
    masked_binary_data = binary_data * observations

    can_query = (targets == 0).astype(int)
    submission = Submission(question_encodings, SAVE_NAME, METHOD)

    for i in range(10):
        print('Feature selection step {}'.format(i+1))
        next_questions = submission.select_feature(masked_data, masked_binary_data, can_query)
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
        submission.update_model(masked_data, masked_binary_data, can_query)

    preds = submission.predict(masked_data, masked_binary_data)

    pred_list = preds[np.where(targets==1)]
    target_list = binary_data[np.where(targets==1)]
    acc = (pred_list == target_list).astype(int).sum()/len(target_list)
    print('Val accuracy: {}'.format(acc))
    return acc

if __name__ == "__main__":
    validate()
