#!/usr/bin/env python
# coding: utf-8

# In[7]:


import hexathello.aiPlayers as aiPlayers
import hexathello.AutoPlayer as autoPlayer
import hexathello.Engine as engine
import hexathello.history as history
import hexathello.jable as jable
import hexathello.printing as printing
import numpy as np
from os import path
from tensorflow.keras.models import load_model
import itertools
import os
import math
from tqdm import tqdm

game_size: int = 5
player_count: int = 2
model_paths = {}

brain_models = []


for i in range(7):
    num_layers = i + 1
    max_neurons = 720
    neurons_per_layer = math.floor(max_neurons / num_layers)
    total_neurons = num_layers * neurons_per_layer

    model_name = "{}x{}_total{}_size{}_players{}_final.keras".format(
        num_layers, neurons_per_layer, total_neurons, game_size, player_count
    )
    full_path = path.join("data", "ai", "vary_dimensions", model_name)
    model_paths[f"path_to_model_{i}"] = full_path
    model = load_model(model_paths[f"path_to_model_{i}"])
    brain_models.append(model)    



ai_keras_0 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[0],
    ai_id="1x720_total720_size5_players2_final", 
    p_random = 0.3
)

ai_keras_1 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[1],
    ai_id="2x360_total720_size5_players2_final", 
    p_random = 0.3
)

ai_keras_2 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[2],
    ai_id="3x240_total720_size5_players2_final", 
    p_random = 0.3
)

ai_keras_3 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[3],
    ai_id="4x180_total720_size5_players2_final", 
    p_random = 0.3
)

ai_keras_4 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[4],
    ai_id="5x144_total720_size5_players2_final", 
    p_random = 0.3
)

ai_keras_5 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[5],
    ai_id="6x120_total720_size5_players2_final", 
    p_random = 0.3
)

ai_keras_6 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[6],
    ai_id="7x102_total714_size5_players2_final", 
    p_random = 0.3
)



# In[8]:


agents_list = [ai_keras_0, ai_keras_1, ai_keras_2, ai_keras_3, ai_keras_4, ai_keras_5, ai_keras_6]

# get elo of each agent in order and update accordingly
current_elo = [4000]*7

# many many games per combo-- let's do this
num_games = 1000

pairs = list(itertools.combinations(agents_list, 2))
total_games = len(pairs) * num_games
pbar = tqdm(total=total_games, desc="Games played")


for agent_i, agent_j in itertools.combinations(agents_list, 2):
    index_i = agents_list.index(agent_i)
    index_j = agents_list.index(agent_j)
    print(f"Running match: {agent_i.ai_id} vs {agent_j.ai_id}")
    agent_i.player_id = 0
    agent_j.player_id = 1
    for runs in range(num_games):
        literalHistory = autoPlayer.runHexathello_withAgents(
            agents=[agent_i, agent_j],
            size=game_size,
            logging_level=0
        )
        if literalHistory._fixed["winner"] == 0:
            winner = index_i
            loser = index_j
            draw = False
        if literalHistory._fixed["winner"] == 1: 
            winner = index_j
            loser = index_i
            draw = False
        if literalHistory._fixed["winner"] == None:
            winner = index_i
            loser = index_j
            draw = True

        current_rating_winner = current_elo[winner]
        current_rating_loser = current_elo[loser]

        expected_winner = 1 / (1 + 10 ** ((current_rating_loser - current_rating_winner) / 400))
        expected_loser = 1 / (1 + 10 ** ((current_rating_winner - current_rating_loser) / 400))
        if not draw: 
            current_elo[winner] = current_elo[winner] + (50*(1-expected_winner))
            current_elo[loser] = current_elo[loser] + (50*(0-expected_loser))
        else:
            current_elo[winner] = current_elo[winner] + (50*(0.5-expected_winner))
            current_elo[loser] = current_elo[loser] + (50*(0.5-expected_loser))
        pbar.update(1)

pbar.close()

print(f"final elos: {current_elo}")

