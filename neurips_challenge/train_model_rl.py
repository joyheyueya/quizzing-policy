'''
a dummy script to "train" a model and save the "trained model"
(just saved an initialized model for local testing) 
'''

import os
import torch
import torch.nn as nn
from model import PyTorchModel
from rl import PolicyNetwork
import pandas as pd
import numpy as np
import torch.optim as optim
from copy import deepcopy
import datetime
import random
import local_evaluation
import warnings
warnings.filterwarnings("ignore")

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

def rand_bin_array(K, N):
    '''
    return an array of K 1's and N-K 0's (randomly shuffled)
    '''
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def split_train_data(df, percentage=0.2):
    '''
    split a fixed percentage of each user's training data into the target and non-target data randomly 
    '''
    df_with_target = pd.DataFrame()
    for u in df['UserId'].unique():
        df_user = df.loc[df.UserId == u]
        df_user['IsTarget'] = rand_bin_array(int(len(df_user)*percentage),len(df_user))
        df_with_target = pd.concat([df_with_target, df_user])
    return df_with_target

def update_policy(rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)

    # if len(discounted_rewards) > 1:
    #     discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    optimizer_rl.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer_rl.step()

if __name__ == "__main__":
    MAX_EPISODE = 1000000
    METHOD = 'rl'
    SAVE_NAME = 'real_' + METHOD + '_volta15_0'
    MAX_STEP = 10
    # VALIDATE = True
    VALIDATE = False
    GAMMA = 1.0

    learning_rate_fc = 3e-3
    learning_rate_rl = 3e-4

    fc_net = PyTorchModel()
    # fc_net.load_state_dict(torch.load('model_task_4_real_volta13_1.pt'))
    # fc_net.load_state_dict(torch.load('fc_net_' + SAVE_NAME + '.pt'))
    # fc_net.load_state_dict(torch.load('fc_net_' + SAVE_NAME + '.pt', map_location=torch.device('cpu')))
    fc_net.load_state_dict(torch.load('fc_net_real_rand_reinforce02_1.pt'))
    fc_net.to(device)
    oprtimizer_fc = optim.Adam(fc_net.parameters(), lr=learning_rate_fc)
    criterion = nn.BCELoss()
    fc_net.train()

    if 'rl' in METHOD:
        rl_net = PolicyNetwork()
        # rl_net.load_state_dict(torch.load('rl_simulation_volta13_0_02-04 11/42.pt'))
        rl_net.load_state_dict(torch.load('rl_net_' + SAVE_NAME + '.pt'))
        # rl_net.load_state_dict(torch.load('rl_net_' + SAVE_NAME + '.pt', map_location=torch.device('cpu')))
        rl_net.to(device)
        optimizer_rl = optim.Adam(rl_net.parameters(), lr=learning_rate_rl)
        rl_net.train()

    train_acc = []
    val_acc= []

    # get all 57 concepts
    concepts_list = []
    subject = pd.read_csv('subject_metadata_indexed.csv')
    answer = pd.read_csv('train_task_3_4_with_concept.csv')
    subject_leaves = subject.loc[subject.Level == 3]
    answer_aggregated = answer.groupby('Concept').mean()
    # delete concepts that are not associated with any questions in the dataset
    old_list = subject_leaves['SubjectIndex'].copy()
    subject_leaves = subject_leaves.set_index('SubjectIndex')
    for o in old_list:
        if o not in answer_aggregated.index:
            subject_leaves = subject_leaves.drop(o) 

    subject_leaves = subject_leaves.reset_index()      
    parents = subject_leaves['ParentIndex'].unique()
    for i in range(len(parents)):
        subject_family = subject_leaves.loc[subject_leaves.ParentIndex == parents[i]]
        for s in subject_family['SubjectIndex']:
            concepts_list.append(s)

    def get_question_meta_indexed(question_df, subject):
        '''
        Returns the a question_df with level 3 subject index
        
        question_df: a dataframe with QuestionId and SubjectId
        subject: the output of get_subject_meta_indexed
        '''
        
        concept_level_3_list_all = []
        for i in range(len(question_df)):
            concept_list = question_df.iloc[i]['SubjectId'][1:-1].split(', ')
            concept_level_3_list = []
            for c in concept_list:
                if subject.loc[subject.SubjectId == int(c)]['Level'].values[0] == 3:
                    concept_level_3_list.append(subject.loc[subject.SubjectId == int(c)]['SubjectIndex'].values[0])
            concept_level_3_list_all.append(concept_level_3_list)
        question_df['SubjectIndexLevelThree'] = concept_level_3_list_all
        return question_df
    question = get_question_meta_indexed(pd.read_csv('data/metadata/question_metadata_task_3_4.csv'), subject)

    question_encodings = []
    # concept_encodings = np.zeros((len(concepts_list), len(question)), dtype=np.float32)
    for i in range(len(question)):
        current_question_concept = question.loc[question.QuestionId == i]['SubjectIndexLevelThree'].values[0][0]
        current_question_concept_index = concepts_list.index(int(current_question_concept))
        current_question_encoding = np.zeros((len(concepts_list), ), dtype=np.float32)
        current_question_encoding[current_question_concept_index] = 1  
        question_encodings.append(current_question_encoding)
        # concept_encodings[current_question_concept_index, i] = 1
    question_encodings = np.array(question_encodings)

    print('started loading data', datetime.datetime.now())
    # this is where the results are saved to
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/models'):
        os.makedirs('results/models')
    if not os.path.exists('results/models/' + SAVE_NAME):
        os.makedirs('results/models/' + SAVE_NAME)
    if not os.path.exists('results/models/' + SAVE_NAME + '/fc_net/'):
        os.makedirs('results/models/' + SAVE_NAME + '/fc_net/')
    if not os.path.exists('results/models/' + SAVE_NAME + '/rl_net/'):
        os.makedirs('results/models/' + SAVE_NAME + '/rl_net/')

    df = pd.read_csv('data/train_data/train_task_3_4.csv')
    df = split_train_data(df)
    data_full = pivot_df(df, 'AnswerValue')
    # print('Number of users:', data.shape)
    binary_data_full = pivot_df(df, 'IsCorrect')  
    # Array containing -1 for unobserved, 0 for observed and not target (can query), 1 for observed and target (held out
    # for evaluation).
    targets_full = pivot_df(df, 'IsTarget')

    # selected_users_val = random.sample(list(range(615)), 20)
    for episode in range(MAX_EPISODE):
        print('started episode ' + str(episode), datetime.datetime.now())
        selected_users = random.sample(list(range(len(data_full))), 1)
        data = data_full[selected_users]
        print('Number of users:', data.shape)
        binary_data = binary_data_full[selected_users]
        targets = targets_full[selected_users]

        observations = np.zeros_like(deepcopy(data))
        masked_data = deepcopy(data) * observations
        masked_binary_data = deepcopy(binary_data) * observations

        can_query = (targets == 0).astype(int)

        # reset mrfs
        fc_net.mrfs = []
        rewards = []
        log_probs = []
        previous_acc = 0

        for s in range(MAX_STEP):
            print('Feature selection step {}'.format(s+1))
            # print(datetime.datetime.now())
            
            if 'rand' in METHOD:
                next_questions = fc_net.select_feature_rand(can_query)
            elif 'us' in METHOD:
                next_questions = fc_net.select_feature_us(can_query, question_encodings)
            elif 'rl' in METHOD:
                action, log_prob, next_questions = fc_net.select_feature_rl(can_query, question_encodings, rl_net)
                log_probs.append(log_prob)

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
            fc_net.update_model(masked_binary_data, next_questions, question_encodings)
            if 'rl' in METHOD:
                # concatenate multiple students' knowledge state into a matrix of shape (num_students, num_concepts)
                predicted_states = []
                for u in range(len(fc_net.mrfs)):
                    predicted_states.append(fc_net.mrfs[u].correct_probs.numpy())
                predicted_states = np.array(predicted_states)
                # print('predicted_states.shape', predicted_states.shape)
                output = fc_net(torch.from_numpy(predicted_states).to(device))
                # print('output.shape', output.shape)               
                preds = (output.detach().cpu().numpy() >= 0.5).astype(int)
                pred_list = preds[np.where(targets==1)]
                target_list = binary_data[np.where(targets==1)]
                current_acc = (pred_list == target_list).astype(int).sum()/len(target_list)
                rewards.append(current_acc - previous_acc)
                previous_acc = current_acc

        if 'rl' in METHOD:
            update_policy(rewards, log_probs)

        print('finished episode ' + str(episode), datetime.datetime.now())
        # oprtimizer_fc.zero_grad()

        # concatenate multiple students' knowledge state into a matrix of shape (num_students, num_concepts)
        predicted_states = []
        for u in range(len(fc_net.mrfs)):
            predicted_states.append(fc_net.mrfs[u].correct_probs.numpy())
        predicted_states = np.array(predicted_states)
        print('predicted_states.shape', predicted_states.shape)
        output = fc_net(torch.from_numpy(predicted_states).to(device))
        print('output.shape', output.shape)
        # loss = criterion(output[np.where(targets==1)], torch.from_numpy(binary_data)[np.where(targets==1)].to(device).float())
        # print(loss)
        # loss.backward()
        # oprtimizer_fc.step()

        preds = (output.detach().cpu().numpy() >= 0.5).astype(int)
        pred_list = preds[np.where(targets==1)]
        target_list = binary_data[np.where(targets==1)]
        acc = (pred_list == target_list).astype(int).sum()/len(target_list)
        currentDT = datetime.datetime.now() 
        print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
        print('Training accuracy: {}'.format(acc))
        train_acc.append(acc)

        if episode % 20 == 0:
            # torch.save(fc_net.state_dict(), 'results/models/' + SAVE_NAME + '/fc_net/fc_net_' + SAVE_NAME + '_' + currentDT.strftime("%m-%d %H:%M") +'.pt')        
            # torch.save(fc_net.state_dict(), 'fc_net_' + SAVE_NAME + '.pt')
            if 'rl' in METHOD:
                torch.save(rl_net.state_dict(), 'results/models/' + SAVE_NAME + '/rl_net/rl_net_' + SAVE_NAME + '_' + currentDT.strftime("%m-%d %H:%M") +'.pt')
                torch.save(rl_net.state_dict(), 'rl_net_' + SAVE_NAME + '.pt')
        
        if VALIDATE:
            print()
            print('started val ' + str(episode), datetime.datetime.now())
            acc = local_evaluation.validate(selected_users_val, question_encodings, SAVE_NAME, METHOD)
            val_acc.append(acc)
        if episode % 20 == 0:
            results = pd.DataFrame()
            results['train_' + METHOD] = train_acc
            if VALIDATE:
                results['val_' + METHOD] = val_acc
            results.to_csv('results/results_' + SAVE_NAME + '.csv', index=False)
        print()
