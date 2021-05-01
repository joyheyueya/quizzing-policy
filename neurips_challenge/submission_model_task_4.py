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
    def __init__(self, question_encodings, SAVE_NAME, METHOD):
        self.num_actions = 2 # query or not query
        self.input_dim_rl = 57*2 # number of (concept, response pair)
        self.hidden_dim_rl = 512


        self.model = PyTorchModel()
        self.model.load_state_dict(torch.load('fc_net_' + SAVE_NAME + '.pt'))
        # self.model.load_state_dict(torch.load('model_task_4_real_volta13_1.pt'))
        self.model.to(device)
        print("Loaded params:")
        for param in self.model.state_dict():
            print(param, "\t", self.model.state_dict()[param].size())
        if 'rl' in METHOD:
            self.rl_net = PolicyNetwork()
            self.rl_net.load_state_dict(torch.load('rl_net_' + SAVE_NAME + '.pt'))
            # self.rl_net.load_state_dict(torch.load('../../../rl_simulation_volta13_0.pt'))
            self.rl_net.to(device)
        self.previous_selections = []
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