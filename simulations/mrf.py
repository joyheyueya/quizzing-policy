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

class MRF():

    def __init__(self, ):
        # a student's true knowledge states (sampled from a distribution based on concept difficulty)
        subject = pd.read_csv('subject_metadata_indexed.csv')
        answer = pd.read_csv('train_task_3_4_with_concept.csv')
        inference = inference_lbp.Inference_lbp(answer, subject, uniform=True)
        self.inference = inference
        self.graph = inference.graph
        knowledge = inference.run_lbp(self.graph, normalize=True)
        self.correct_probs = get_correct_probs(knowledge)

        # index = 0
        # self.index_to_concept = dict()
        # for k in knowledge:
        #     self.index_to_concept[index] = k[0].name
        #     index = index + 1
        index = 0
        self.index_to_unary_factor = dict()
        for i in range(len(self.graph._factors)):
            rv_list = self.graph._factors[i]._rvs
            if len(rv_list) == 1:
                self.index_to_unary_factor[index] = self.graph._factors[i]
                index = index + 1

    def update_graph(self, concept, response):
        '''
        Args:
            concept (int): the index of a concept      
        '''
        old_correct = self.index_to_unary_factor[concept]._potential[1]
        if response:
            new_correct = old_correct + 0.2
        else:
            new_correct = old_correct - 0.2
        if new_correct < 0:
            new_correct = 0.0
        if new_correct > 1:
            new_correct = 1.0
        # if response:
        #     new_correct = 1.0
        # else:
        #     new_correct = 0.0
        self.index_to_unary_factor[concept]._potential = np.array([1-new_correct, new_correct])

        # rerun lbp
        knowledge = self.inference.run_lbp(self.graph, normalize=True)
        self.correct_probs = get_correct_probs(knowledge)
        return (self.correct_probs >= 0.5).float()


                
        # concept = self.index_to_concept[concept]
        # for i in range(len(self.graph._factors)):
        #     rv_list = self.graph._factors[i]._rvs
        #     if len(rv_list) == 1:
        #         # print(rv_list[0].name)
        #         if rv_list[0].name == concept:
        #             # print('rv_list[0].name')
        #             old_incorrect = self.graph._factors[i]._potential[0]
        #             if response:
        #                 new_incorrect = old_incorrect - 0.2
        #             else:
        #                 new_incorrect = old_incorrect + 0.2
        #             if new_incorrect < 0:
        #                 new_incorrect = 0.0
        #             if new_incorrect > 1:
        #                 new_incorrect = 1.0
        #             self.graph._factors[i]._potential = np.array([new_incorrect, 1 - new_incorrect])
        #             # self.graph._factors[i]._potential = np.array([float(1-response), float(response)])
        #             break
        # # rerun lbp
        # knowledge = self.inference.run_lbp(self.graph, normalize=True)
        # self.incorrect_probs = get_incorrect_probs(knowledge)
        # return (self.incorrect_probs<=0.5).float()



