#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# In[6]:


agents_list = [ai_keras_0, ai_keras_1, ai_keras_2, ai_keras_3, ai_keras_4, ai_keras_5, ai_keras_6]

save_dir = path.join("data", "history", "matchups_final")
os.makedirs(save_dir, exist_ok=True)

# update file for new data
combined_path = path.join(save_dir, "all_data_final.json")

if path.isfile(combined_path):
    combined_history: jable.JyFrame = jable.read_file(combined_path)
else:
    combined_history: jable.JyFrame = history.new_literalHistory(
        player_count=player_count,
        size=game_size
    )
    combined_history = history.history_asInt(combined_history)
    combined_history.makeColumn_shift("winner")
    combined_history.makeColumn_shift("scores")

num_games = 100

for agent_i, agent_j in itertools.combinations(agents_list, 2):
    print(f"Running match: {agent_i.ai_id} vs {agent_j.ai_id}")
    agent_i.player_id = 0
    agent_j.player_id = 1
    for runs in range(num_games):
        literalHistory = autoPlayer.runHexathello_withAgents(
            agents=[agent_i, agent_j],
            size=game_size,
            logging_level=0
        )

        chunk = history.history_asInt(literalHistory)

        combined_history.extend(chunk)
    combined_history.write_file(fp=combined_path)


print("All matchups recorded to:", combined_path)
print(f"After {num_games} games per pairing, total moves: {len(combined_history)}")


# In[5]:


# in `quickstart_recording_data` We'll look at saving many games to disk to use for machine learning

