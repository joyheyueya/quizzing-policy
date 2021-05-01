import os
import numpy as np
import torch
from model import PyTorchModel
import pandas as pd
from rl import PolicyNetwork

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    # print("GPU is available")
else:
    device = torch.device("cpu")

class Submission:
    """
    API Wrapper class which loads a saved model upon construction, and uses this to implement an API for feature 
    selection and missing value prediction.
    """
    def __init__(self, SAVE_NAME, METHOD):
        print('SAVE_NAME', SAVE_NAME)
        print('METHOD', METHOD)
        self.num_actions = 2 # query or not query
        self.input_dim_rl = 57*2 # number of (concept, response pair)
        self.hidden_dim_rl = 512


        self.model = PyTorchModel()
        self.model.load_state_dict(torch.load('fc_net_real_rand_reinforce02_1.pt'))
        # self.model.load_state_dict(torch.load('fc_net_' + SAVE_NAME + '.pt', map_location=torch.device('cpu')))
        # self.model.load_state_dict(torch.load('model_task_4_real_volta13_1.pt'))
        self.model.to(device)
        print("Loaded params:")
        for param in self.model.state_dict():
            print(param, "\t", self.model.state_dict()[param].size())
        if 'rl' in METHOD:
            self.rl_net = PolicyNetwork()
            self.rl_net.load_state_dict(torch.load('rl_net_' + SAVE_NAME + '.pt'))
            # self.rl_net.load_state_dict(torch.load('rl_net_' + SAVE_NAME + '.pt', map_location=torch.device('cpu')))
            # self.rl_net.load_state_dict(torch.load('../../../rl_simulation_volta13_0.pt'))
            self.rl_net.to(device)
        self.previous_selections = []

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

        self.question_encodings = question_encodings
        self.method = METHOD

    def select_feature(self, masked_data, masked_binary_data, can_query):
        """
        Use your model to select a new feature to observe from a list of candidate features for each student in the
            input data, with the goal of selecting features with maximise performance on a held-out set of answers for
            each student.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing data revealed to the model
                at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """
        # Use the loaded model to perform feature selection.
        if 'rand' in self.method:
            selections = self.model.select_feature_rand(can_query)
        elif 'us' in self.method:
            selections = self.model.select_feature_us(can_query, self.question_encodings)
        elif 'rl' in self.method:
            _, _, selections = self.model.select_feature_rl(can_query, self.question_encodings, self.rl_net)
        self.previous_selections = selections

        return selections

    def update_model(self, masked_data, masked_binary_data, can_query):
        """
        Update the model to incorporate newly revealed data if desired (e.g. training or updating model state).
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing categorical data revealed to 
                the model at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """
        # Update the model after new data has been revealed, if required.
        self.model.update_model(masked_binary_data, self.previous_selections, self.question_encodings)

    def predict(self, masked_data, masked_binary_data):
        """
        Use your model to predict missing binary values in the input data. Both categorical and binary versions of the
        observed input data are available for making predictions with.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing categorical data revealed to 
                the model at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
        Returns:
            predictions (np.array): Array of shape (num_students, num_questions) containing predictions for the
                unobserved values in `masked_binary_data`. The values given to the observed data in this array will be 
                ignored.
        """
        # Use the loaded model to perform missing value prediction.
        # concatenate multiple students' knowledge state into a matrix of shape (num_students, num_concepts)
        predicted_states = []
        for u in range(len(self.model.mrfs)):
            predicted_states.append(self.model.mrfs[u].correct_probs.numpy())
        predicted_states = np.array(predicted_states)

        predictions = self.model.predict(torch.from_numpy(predicted_states).to(device))

        return predictions
