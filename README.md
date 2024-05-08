# SSH-RLKEX

This is the GitHub Repository for my master thesis: "Utilizing Reinforcement Learning for SSH Key Retrieval using Graph".

The code used to train and test the model in the thesis is in the commit https://github.com/CyrilGomes/RLKEX/commit/cbe1cd8d46d8f5c301fd0e241eeec8a2538802f9

The folder Graph_Gen contains the rust project to convert heap dumps into graphs.

The folder Graph_insights contains codes used to better understand the nature of the generated graphs.

The Heuristics Folder contains some code to test and train the nb_keys_classifier as well as the root predictor.
(nb_keys_classifier.ipynb and root_heuristic.py is used for training)

The folder RL/GNNDRL contains code used to test Graph Neural Networks with Deep Reinforcement Learning,
The method we chose to use is contained in "variable_action_space_goal"

