# quizzing-policy

This repo is for the paper: Quizzing Policy Using Reinforcement Learning for Inferring the Student Knowledge State (EDM '21)

## Preparing data

Download the [Eedi dataset](https://eedi.com/projects/neurips-education-challenge), and put the `data` folder inside the `neurips_challenge` directory.

## Simulations

Run `al.py` from the `simulations` directory.

## The Neurips 2020 Education Challenge

First, run `train_model_rand.py` from the `neurips_challenge` directory to train a model for predicitng students' responses based on beliefs about their knowledge states.

Then, run `train_model_rl.py` from the `neurips_challenge` directory to train QP-RL.

To evaluate different models, run `evaluation.py` from the `neurips_challenge` directory. Change the `METHOD` constant for switching models.
