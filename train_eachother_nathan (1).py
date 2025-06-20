#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


for i in range(9):
    num_layers = i + 1
    neurons_per_layer = 122
    total_neurons = num_layers * neurons_per_layer

    model_name = "kha_layers_{}.keras".format(
        num_layers
    )
    full_path = path.join("data", "ai", "layer_sweeper", model_name)
    model_paths[f"path_to_model_{i}"] = full_path
    model = load_model(model_paths[f"path_to_model_{i}"])
    brain_models.append(model)    



ai_keras_0 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[0],
    ai_id="kha_layers_1", 
    p_random = 0.3
)

ai_keras_1 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[1],
    ai_id="kha_layers_2", 
    p_random = 0.3
)


ai_keras_2 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[2],
    ai_id="kha_layers_3", 
    p_random = 0.3
)

ai_keras_3 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[3],
    ai_id="kha_layers_4", 
    p_random = 0.3
)


ai_keras_4 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[4],
    ai_id="kha_layers_5", 
    p_random = 0.3
)



ai_keras_5 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[5],
    ai_id="kha_layers_6", 
    p_random = 0.3
)


ai_keras_6 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[6],
    ai_id="kha_layers_7", 
    p_random = 0.3
)

ai_keras_7 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[7],
    ai_id="kha_layers_8", 
    p_random = 0.3
)

ai_keras_8 = aiPlayers.KerasHexAgent(
    size=game_size,
    player_count=player_count,
    brain=brain_models[8],
    ai_id="kha_layers_9", 
    p_random = 0.3
)




# In[6]:


agents_list = [ai_keras_0, ai_keras_1, ai_keras_2, ai_keras_3, ai_keras_4, ai_keras_5, ai_keras_6, ai_keras_7, ai_keras_8]

save_dir = path.join("data", "history", "matchups_layers")
os.makedirs(save_dir, exist_ok=True)

# update file for new data
combined_path = path.join(save_dir, "all_matchups_manydata.json")

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

num_games = 50

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
        print(combined_path)
        combined_history.write_file(fp=combined_path)


print("All matchups recorded to:", combined_path)
print(f"After {num_games} games per pairing, total moves: {len(combined_history)}")


# In[5]:


# in `quickstart_recording_data` We'll look at saving many games to disk to use for machine learning

