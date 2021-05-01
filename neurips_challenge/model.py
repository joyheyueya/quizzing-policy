import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from mrf import MRF
from copy import deepcopy
from rl import PolicyNetwork
import datetime

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def get_action_uncertainty_sampling(state):
    min_diff_from_cutoff = torch.min((state-0.5).abs())
    # return int(torch.argmin((state-0.5).abs()))
    return int(np.random.choice(np.where((state-0.5).abs() == min_diff_from_cutoff)[0]))


class PyTorchModel(nn.Module):
    """
    Simple example model to illustrate saving and loading.
    """
    def __init__(self):
        super(PyTorchModel, self).__init__()
        # self.fc1 = nn.Linear(308,1024)
        # self.fc2 = nn.Linear(1024,512)
        # self.fc3 = nn.Linear(512,948)
        self.fc1 = nn.Linear(57,256)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,948)
        self.sigmoid = nn.Sigmoid()
        # self.drop = torch.nn.Dropout(0.2)
        self.mrfs = []
        # self.rl_net = None
        # self.rl_lr = 3e-4

    def forward(self, x):
        x = self.fc1(x)
        # x = drop(x)
        x = F.relu(x)

        x = self.fc2(x)
        # x = drop(x)
        x = F.relu(x)  
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x     

    def predict(self, knowledge_state):
        output = self.forward(knowledge_state)
        # treat the output as the probabilities of correctness
        preds = (output.detach().cpu().numpy() >= 0.5).astype(int)
        return preds

    def select_feature_rand(self, can_query):
        selections = []
        for i in range(can_query.shape[0]):
            can_query_row = can_query[i,:]
            selected_feature = np.random.choice(np.where(can_query_row==1)[0])
            selections.append(selected_feature)
            # print('finished s', str(i))
        return selections

    def select_feature_us(self, can_query, question_encodings):
        if len(self.mrfs) == 0:
            self.mrfs.append(MRF())
            for i in range(can_query.shape[0]-1):
                # print('initialized mrf', str(i))
                self.mrfs.append(deepcopy(self.mrfs[0]))

        selections = []
        for i in range(can_query.shape[0]):
            can_query_row = can_query[i,:]
            can_query_questions = np.where(can_query_row==1)[0]
            can_query_concepts = np.where(np.sum(question_encodings[can_query_questions], axis=0))[0]
            selected_can_query_concept_index = get_action_uncertainty_sampling(self.mrfs[i].correct_probs[can_query_concepts])
            selected_can_query_concept = can_query_concepts[selected_can_query_concept_index]
            can_query_questions_for_selected_concept = question_encodings[can_query_questions][:, selected_can_query_concept]
            selected_query_question_index = np.random.choice(np.where(can_query_questions_for_selected_concept)[0])
            selected_feature = can_query_questions[selected_query_question_index]
            selections.append(selected_feature)
            # print('finished s', str(i))
        return selections

    def select_feature_rl(self, can_query, question_encodings, rl_net):
        if len(self.mrfs) == 0:
            self.mrfs.append(MRF())
            for i in range(can_query.shape[0]-1):
                # print('initialized mrf', str(i))
                self.mrfs.append(deepcopy(self.mrfs[0]))
        # print('mrf', self.mrfs[0].correct_probs)
        # if self.rl_net == None:
        #     self.rl_net = PolicyNetwork(57, 57)
        #     # self.rl_net.load_state_dict(torch.load('rl_simulation_volta13_0_02-04 11/42.pt'))
        #     self.rl_net.to(device)
        #     self.optimizer_rl = optim.Adam(self.rl_net.parameters(), lr=self.rl_lr)
        #     self.rl_net.train()

        # concatenate multiple students' knowledge state into a matrix of shape (num_students, num_concepts)
        predicted_states = []
        for u in range(len(self.mrfs)):
            predicted_states.append(self.mrfs[u].correct_probs.numpy())
        predicted_states = np.array(predicted_states)

        action, log_prob, selections = rl_net.get_action(torch.from_numpy(predicted_states).to(device), can_query, question_encodings)
        # print('select', selections)
        # print()
        return action, log_prob, selections
    
    def update_model(self, masked_binary_data, selections, question_encodings):
        if len(self.mrfs) == 0:
            self.mrfs.append(MRF())
            for i in range(masked_binary_data.shape[0]-1):
                # print('initialized mrf', str(i))
                self.mrfs.append(deepcopy(self.mrfs[0]))
        # print(selections)
        for i in range(masked_binary_data.shape[0]):
            # print('update mrf', str(i))
            # print(datetime.datetime.now())
            # print('np.where(question_encodings[selections[i]])[0][0]', np.where(question_encodings[selections[i]])[0][0])
            # print('masked_binary_data[i, selections[i]]', masked_binary_data[i, selections[i]])
            # print()
            self.mrfs[i].update_graph(np.where(question_encodings[selections[i]])[0][0], masked_binary_data[i, selections[i]])

# class PyTorchModel(nn.Module):
#     """
#     Simple example model to illustrate saving and loading.
#     """
#     def __init__(self):
#         super(PyTorchModel, self).__init__()
#         self.fc1 = nn.Linear(10,20)
#         self.fc2 = nn.Linear(20,10)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

#     def predict(self, masked_data, masked_binary_data):
#         predictions = np.zeros_like(masked_binary_data)
#         predictions = np.random.choice([0,1], size=masked_binary_data.shape)
#         return predictions

#     def select_feature(self, masked_data, can_query):
#         selections = []
#         for i in range(masked_data.shape[0]):
#             can_query_row = can_query[i,:]
#             selected_feature = np.random.choice(np.where(can_query_row==1)[0])
#             selections.append(selected_feature)
#         return selections