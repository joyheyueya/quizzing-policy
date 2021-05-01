import pandas as pd
import math
import numpy as np
import factorgraph as fg
import datetime
import itertools

class Inference_lbp():

    def __init__(self, answer, subject, unary_correctness=0.5, binary_influence=0.7, uniform=False):
        subject_leaves = subject.loc[subject.Level == 3]
        # print('len(subject_leaves)', len(subject_leaves))
        # calculate the average correctness for each concept
        answer_aggregated = answer.groupby('Concept').mean()
        # print('len(answer_aggregated)', len(answer_aggregated))
        # delete concepts that are not associated with any questions in the dataset
        old_list = subject_leaves['SubjectIndex'].copy()
        subject_leaves = subject_leaves.set_index('SubjectIndex')
        for o in old_list:
            if o not in answer_aggregated.index:
                subject_leaves = subject_leaves.drop(o)  
        # print('len(subject_leaves)', len(subject_leaves))
        subject_leaves = subject_leaves.reset_index()      
        parents = subject_leaves['ParentIndex'].unique()

        # Make an empty graph
        g = fg.Graph()

        # initialize an mrf where all nodes have the same potential
        if uniform:
            for i in range(len(parents)):
                subject_family = subject_leaves.loc[subject_leaves.ParentIndex == parents[i]]
                for s in subject_family['SubjectIndex']:
                    g.rv(str(s), 2) 
                    g.factor([str(s)], potential=np.array([0.5, 0.5]))

                # Add binary factors    
                pairs = list(itertools.combinations(list(subject_family['SubjectIndex']), 2))
                for p in pairs:
                    g.factor([str(p[0]), str(p[1])], potential=np.array([
                            [binary_influence, 1-binary_influence],
                            [1-binary_influence, binary_influence],
                    ]))
        # create a concept difficulty dict (concept -> difficulty) if concept exists in data
        else:
            concept_to_correctness = dict()
            # answer_aggregated = answer.groupby('Concept').mean()
            for c in answer_aggregated.index:
                concept_to_correctness[c] = answer_aggregated.loc[c]['IsCorrect']

            # add each family of concepts to the graph
            for i in range(len(parents)):
            # for i in range(2):
                subject_family = subject_leaves.loc[subject_leaves.ParentIndex == parents[i]]
                # print('len(subject_family)', len(subject_family))
                # Add random variables (RVs)
                for s in subject_family['SubjectIndex']:
                    g.rv(str(s), 2)    
                    if s in concept_to_correctness:
                        # Add unary factors
                        # print(s)
                        # print(concept_to_correctness[s])
                        g.factor([str(s)], potential=np.array([1 - concept_to_correctness[s], concept_to_correctness[s]]))
                    else:
                        g.factor([str(s)], potential=np.array([unary_correctness, 1-unary_correctness]))
                # Add binary factors    
                pairs = list(itertools.combinations(list(subject_family['SubjectIndex']), 2))
                for p in pairs:
                    g.factor([str(p[0]), str(p[1])], potential=np.array([
                            [binary_influence, 1-binary_influence],
                            [1-binary_influence, binary_influence],
                    ]))

        self.graph = g

    def run_lbp(self, graph, normalize=False):

        # Run (loopy) belief propagation (LBP)
        iters, converged = graph.lbp(normalize=True)
        # print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
        # print()
        return graph.print_rv_marginals(normalize=normalize)

    # def update_graph(self, concept, response):
    #     print('inf')
    #     self.graph.factor([str(concept)], potential=np.array([1-response, response]))
