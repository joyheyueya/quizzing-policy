import torch
import torch.nn as nn
import inference_lbp
import pandas as pd
from rl import PolicyNetwork
from student import Student
from mrf import MRF
import torch.optim as optim
import numpy as np
import pandas as pd
import datetime
import sys
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

print("PyTorch:\t{}".format(torch.__version__))
is_cuda = torch.cuda.is_available()
print('is_cuda', is_cuda)
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

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

    optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()

def get_accuracy(predicted, true):
    incorrect_counts = (predicted-true).abs().sum()
    return (len(predicted) - incorrect_counts) / float(len(predicted))


def get_action_uncertainty_sampling(state):
    min_diff_from_cutoff = torch.min((state-0.5).abs())
    # return int(torch.argmin((state-0.5).abs()))
    return int(np.random.choice(np.where((state-0.5).abs() == min_diff_from_cutoff)[0]))

def get_action_rand(state):
    return np.random.choice(len(state))

if __name__ == "__main__":
    MAX_EPISODE = 1000000000
    SAVE_NAME = 'volta13_0_simu_final_diff'
    MAX_STEP = 10
    GAMMA = 1.0
    # plot_episode_avg = 2

    learning_rate = 3e-4
    num_concepts = 57
    input_dim_rl = num_concepts    # number of concepts in the knowledge state
    num_actions = num_concepts

    rl_net = PolicyNetwork(input_dim_rl, num_actions)
    rl_net.to(device)    
    optimizer = optim.Adam(rl_net.parameters(), lr=learning_rate)
    rl_net.train()

    student = Student()
    # student.get_new_student()

    all_rewards = []
    all_rewards_us = []
    all_rewards_rand = []
    avg_all_rewards = []
    avg_all_rewards_us = []
    avg_all_rewards_rand = []

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/models'):
        os.makedirs('results/models')
    if not os.path.exists('results/models/' + SAVE_NAME):
        os.makedirs('results/models/' + SAVE_NAME)

    for episode in range(MAX_EPISODE):
        mrf = MRF()
        # mrf_us = MRF()
        # mrf_rand = MRF()
        mrf_us = deepcopy(mrf)
        mrf_rand = deepcopy(mrf)
        student = Student()
        # print('episode', episode)
        # print(student.student_correct_probs)
        # print(student.knowledge_probs)
        # print(student.knowledge_states)
        # print()

        log_probs = []
        rewards = []
        rewards_us = []
        rewards_rand = []
        previous_acc = 0
        previous_acc_us = 0
        previous_acc_rand = 0

        for step in range(MAX_STEP):
            action, log_prob = rl_net.get_action(mrf.correct_probs.to(device))
            response = student.get_student_response(action)
            predicted_states = mrf.update_graph(action, response)
            current_acc = get_accuracy(predicted_states, student.knowledge_states)
            rewards.append(current_acc - previous_acc)
            previous_acc = current_acc
            log_probs.append(log_prob)

            action_us = get_action_uncertainty_sampling(mrf_us.correct_probs)
            response_us = student.get_student_response(action_us)
            predicted_states_us = mrf_us.update_graph(action_us, response_us)
            current_acc_us = get_accuracy(predicted_states_us, student.knowledge_states)
            rewards_us.append(current_acc_us - previous_acc_us)
            previous_acc_us = current_acc_us

            action_rand = get_action_rand(mrf_rand.correct_probs)
            response_rand = student.get_student_response(action_rand)
            predicted_states_rand = mrf_rand.update_graph(action_rand, response_rand)
            current_acc_rand = get_accuracy(predicted_states_rand, student.knowledge_states)
            rewards_rand.append(current_acc_rand - previous_acc_rand)
            previous_acc_rand = current_acc_rand

        update_policy(rewards, log_probs)

        all_rewards.append(np.sum(rewards))
        # print('all_rewards', all_rewards)
        # if episode % (MAX_EPISODE * 1) == 0:
        #     avg_all_rewards.append(np.mean(all_rewards[-plot_episode_avg:]))

        all_rewards_us.append(np.sum(rewards_us))
        # print('all_rewards_us', all_rewards_us)
        # if episode % (MAX_EPISODE * 1) == 0:
        #     avg_all_rewards_us.append(np.mean(all_rewards_us[-plot_episode_avg:]))

        all_rewards_rand.append(np.sum(rewards_rand))
        # print('all_rewards_rand', all_rewards_rand)
        # if episode % (MAX_EPISODE * 1) == 0:
        #     avg_all_rewards_rand.append(np.mean(all_rewards_rand[-plot_episode_avg:]))

        if episode % 50 == 0:
            currentDT = datetime.datetime.now() 
            print('episode', episode)
            print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
            print('all_rewards_rl', all_rewards[-1])
            print('all_rewards_us', all_rewards_us[-1])
            print('all_rewards_rand', all_rewards_rand[-1])
            print()
            # sys.stdout.write("episode: {}, total reward: {}, average_reward of 10 episodes: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3))) 

            results = pd.DataFrame()
            results['RL'] = all_rewards
            results['US'] = all_rewards_us
            results['random'] = all_rewards_rand
            results.to_csv('results/results_' + SAVE_NAME + '.csv', index=False)
        if episode % 50  == 0:
            torch.save(rl_net.state_dict(), 'rl_simulation_' + SAVE_NAME + '.pt')
            torch.save(rl_net.state_dict(), 'results/models/' + SAVE_NAME + '/rl_simulation_' + SAVE_NAME + '_' + currentDT.strftime("%m-%d %H:%M") +'.pt')

# print('len(avg_all_rewards)', len(avg_all_rewards))
# print('len(avg_all_rewards_us)', len(avg_all_rewards_us))
# print('len(avg_all_rewards_rand)', len(avg_all_rewards_rand))
print('len(all_rewards)', len(all_rewards))
print('len(all_rewards_us)', len(all_rewards_us))
print('len(all_rewards_rand)', len(all_rewards_rand))

plt.plot(all_rewards_rand)
plt.plot(all_rewards_us)
plt.plot(all_rewards)
plt.legend(['Random', 'Uncertainty sampling', 'RL'])
# plt.legend(['RL average number of steps', 'Oracle average number of steps', 'Heuristics average number of steps'])
plt.xlabel('Number of episodes')
plt.ylabel('Final ccuracy')
# plt.title('Average reward of the last ' + str(plot_episode_avg) + ' episodes (normalized)')
plt.savefig('results/test' + str(MAX_EPISODE) + '_' + str(MAX_STEP) + '_' + SAVE_NAME + '.pdf')
plt.show()

