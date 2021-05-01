import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np  
from copy import deepcopy

class PolicyNetwork(nn.Module):
    '''
    num_inputs: the length of the feature respresentation of the states
    '''
    def __init__(self, num_inputs=57, num_actions=57):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.softmax(self.linear2(x), dim=1)
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        
        return x 
    
    def get_action(self, state, can_query, question_encodings):
        # assuming state is in shape [number of features]
        # print('state.shape before', state.shape)
        # reshape state to [1, number of features]
        # state = state.unsqueeze(0)
        # print('state.shape after', state.shape)
        probs = self.forward(Variable(state))
        selections = []
        if probs.shape[0] == 1:
            # probabilities = deepcopy(probs.detach().cpu().numpy())
            # print('probs[0].shape', probs[0].shape)
    #         print()
            # print('np.squeeze(probs.detach().cpu().numpy()).shape', np.squeeze(probs.detach().cpu().numpy()).shape)
            # print('np.squeeze(probs.detach().cpu().numpy())', np.squeeze(probs.detach().cpu().clone().numpy()))
            # print()
            # print('type(probs)', type(probs))
            # print('probs.detach().squeeze().shape', probs.detach().squeeze().shape)
            # print('probs.detach().squeeze()', probs.detach().squeeze())
            # print(probs.detach().squeeze().sum())
            # print('probs', probs)
            # print()
            # probs = probs.to(torch.device("cuda"))
            can_query_row = can_query[0,:]
            # print('can_query_row', can_query_row)
            can_query_questions = np.where(can_query_row==1)[0]
            # print('can_query_questions', can_query_questions)
            can_query_concepts = np.where(np.sum(question_encodings[can_query_questions], axis=0))[0]
            # print('can_query_concepts', can_query_concepts)
            # can_query_concepts_probs = deepcopy(np.squeeze(probs[i].detach().cpu().numpy()[can_query_concepts]))
            can_query_concepts_probs = np.squeeze(probs.detach().cpu().numpy())[can_query_concepts]
            # print('can_query_concepts_probs', can_query_concepts_probs)
            # can_query_concepts_probs = can_query_concepts_probs/can_query_concepts_probs.sum()
            # print('can_query_concepts.shape', can_query_concepts.shape)
            if can_query_concepts.shape[0] > 1:
                selected_can_query_concept_index = np.random.choice(can_query_concepts.shape[0], p=can_query_concepts_probs/can_query_concepts_probs.sum())
            else:
                selected_can_query_concept_index = 0
            # print('selected_can_query_concept_index', selected_can_query_concept_index)
            selected_can_query_concept = can_query_concepts[selected_can_query_concept_index]
            # print('selected_can_query_concept', selected_can_query_concept)
            highest_prob_action = selected_can_query_concept
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
            # print('log_prob', log_prob)
            can_query_questions_for_selected_concept = question_encodings[can_query_questions][:, selected_can_query_concept]
            # print('can_query_questions_for_selected_concept', can_query_questions_for_selected_concept)
            selected_query_question_index = np.random.choice(np.where(can_query_questions_for_selected_concept)[0])
            # print('selected_query_question_index', selected_query_question_index)
            selected_feature = can_query_questions[selected_query_question_index]
            # print('selected_feature', selected_feature)
            selections.append(selected_feature)
        else:
            # for validation
            highest_prob_action = None
            log_prob = None
            probabilities = probs.detach().cpu().numpy()
            for i in range(can_query.shape[0]):
                can_query_row = can_query[i,:]
                can_query_questions = np.where(can_query_row==1)[0]
                can_query_concepts = np.where(np.sum(question_encodings[can_query_questions], axis=0))[0]
                # can_query_concepts_probs = deepcopy(np.squeeze(probs[i].detach().cpu().numpy()[can_query_concepts]))
                can_query_concepts_probs = probabilities[i][can_query_concepts]
                if can_query_concepts.shape[0] > 1:
                    selected_can_query_concept_index = np.random.choice(can_query_concepts.shape[0], p=can_query_concepts_probs/can_query_concepts_probs.sum())
                else:
                    selected_can_query_concept_index = 0
                selected_can_query_concept = can_query_concepts[selected_can_query_concept_index]
                can_query_questions_for_selected_concept = question_encodings[can_query_questions][:, selected_can_query_concept]
                selected_query_question_index = np.random.choice(np.where(can_query_questions_for_selected_concept)[0])
                selected_feature = can_query_questions[selected_query_question_index]
                selections.append(selected_feature)
        
        return highest_prob_action, log_prob, selections

    def get_action_val(self, state, can_query, question_encodings):
        # assuming state is in shape [number of features]
        # print('state.shape before', state.shape)
        # reshape state to [1, number of features]
        # state = state.unsqueeze(0)
        # print('state.shape after', state.shape)
        probs = self.forward(Variable(state))
        probabilities = deepcopy(probs.detach().cpu().numpy())
        # print('probs[0].shape', probs[0].shape)
#         print()
        # print('np.squeeze(probs.detach().cpu().numpy()).shape', np.squeeze(probs.detach().cpu().numpy()).shape)
        # print('np.squeeze(probs.detach().cpu().numpy())', np.squeeze(probs.detach().cpu().clone().numpy()))
        # print()
        # print('type(probs)', type(probs))
        # print('probs.detach().squeeze().shape', probs.detach().squeeze().shape)
        # print('probs.detach().squeeze()', probs.detach().squeeze())
        # print(probs.detach().squeeze().sum())
        # print('probs', probs)
        # print()
        # probs = probs.to(torch.device("cuda"))
        selections = []
        highest_prob_action = []
        log_prob = []
        for i in range(can_query.shape[0]):
            can_query_row = can_query[i,:]
            can_query_questions = np.where(can_query_row==1)[0]
            can_query_concepts = np.where(np.sum(question_encodings[can_query_questions], axis=0))[0]
            # can_query_concepts_probs = deepcopy(np.squeeze(probs[i].detach().cpu().numpy()[can_query_concepts]))
            can_query_concepts_probs = deepcopy(probabilities[i][can_query_concepts])
            # can_query_concepts_probs = can_query_concepts_probs/can_query_concepts_probs.sum()
            # print('can_query_concepts', can_query_concepts)
            # print('can_query_concepts.shape', can_query_concepts.shape)
            if can_query_concepts.shape[0] > 1:
                selected_can_query_concept_index = np.random.choice(can_query_concepts.shape[0], p=can_query_concepts_probs/can_query_concepts_probs.sum())
            else:
                selected_can_query_concept_index = 0
            selected_can_query_concept = can_query_concepts[selected_can_query_concept_index]
            highest_prob_action.append(selected_can_query_concept)
            log_prob.append(torch.log(probs[i][highest_prob_action]))
            can_query_questions_for_selected_concept = question_encodings[can_query_questions][:, selected_can_query_concept]
            selected_query_question_index = np.random.choice(np.where(can_query_questions_for_selected_concept)[0])
            selected_feature = can_query_questions[selected_query_question_index]
            selections.append(selected_feature)

        # highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().cpu().numpy()))
        # log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        
        return highest_prob_action, log_prob, selections
