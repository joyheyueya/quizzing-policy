import pandas as pd
import inference_lbp
import numpy as np
import torch

def get_correct_probs(knowledge):
    correct_probs = []
    for k in knowledge:
        # store correct probabilities in a list
        correct_probs.append(k[1][1])
    return torch.tensor(correct_probs, dtype=torch.float)

class Student():

    def __init__(self, ):
        # a student's true knowledge states (sampled from a distribution based on concept difficulty)
        subject = pd.read_csv('subject_metadata_indexed.csv')
        answer = pd.read_csv('train_task_3_4_with_concept.csv')
        inference = inference_lbp.Inference_lbp(answer, subject)
        self.graph = inference.graph
        knowledge = inference.run_lbp(self.graph, normalize=True)
        self.student_correct_probs = get_correct_probs(knowledge)
        # self.knowledge_states = (self.student_correct_probs >= 0.5).float()

        knowledge_probs = []
        # index = 0
        # self.concept_to_index = dict()
        for k in knowledge:
            knowledge_probs.append(k[1])
            # self.concept_to_index[k[0].name] = index
            # index = index + 1
        self.knowledge_probs = np.array(knowledge_probs)
        weights = torch.tensor(self.knowledge_probs, dtype=torch.float)
        self.knowledge_states = torch.multinomial(weights, 1, replacement=True).squeeze()

    # def get_new_student(self, ):
    #     weights = torch.tensor(self.knowledge_probs, dtype=torch.float)
    #     self.knowledge_states = torch.multinomial(weights, 1, replacement=True).squeeze()

    def get_student_response(self, concept):
        '''
        Args:
            concept (int)
        '''
        return self.knowledge_states[concept]
        # return self.knowledge_states[self.concept_to_index[str(concept)]]
