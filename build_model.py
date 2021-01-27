#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:20:07 2021

@author: en-chengchang
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os, time

# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 

min_n_nodes = 4
max_n_nodes = 7
n_attached = 2
p_dim = 4
T = 4

max_epi = 1
min_epi = 0.5
cur_epi = max_epi
decay_epi = 0.99

gamma = 0.9

n_steps = 2
train_data = []
n_replay = 20

batch = 32
    
# ====================================================== # 
#                                                        #
#                                                        #
#                                                        #
# ====================================================== # 

def generate_graph(min_n_nodes=min_n_nodes, max_n_nodes=max_n_nodes , 
                   n_attached=n_attached, show_pic = True):
    g = nx.barabasi_albert_graph(random.randint(min_n_nodes, max_n_nodes),n_attached)
    edges = list(g.edges())
    g_link = g._adj
    n_nodes = len(g_link)
    if show_pic: 
        nx.draw(g, with_labels=True, font_weight='bold')
        plt.show()
    return g, edges, g_link, n_nodes
    
# ====================================================== # 
#                                                        #
#       model                                            #
#                                                        #
# ====================================================== # 

def build_model(p_dim=p_dim):
    # x_embedding model 
    input_x = layers.Input(shape=(2,), name = 'x')
    layer_x = layers.Dense(p_dim, activation='sigmoid')(input_x)
    
    # u_embedding_sum model 
    input_u = layers.Input(shape=(p_dim,), name = 'u')
    layer_u = layers.Dense(p_dim, activation='sigmoid')(input_u)
    
    # concate1
    updated_layer = layers.Add()([layer_x, layer_u])
    updated_layer = layers.Dense(p_dim, activation='relu')(updated_layer)
    embedding = keras.Model(inputs=[input_x, input_u], outputs=updated_layer, name="embed")
    # embedding.summary()
    
    input_sum_u = layers.Input(shape=(p_dim * 2,), name = 's')
    layer_final = layers.Dense(p_dim * 2, activation='sigmoid')(input_sum_u)
    layer_final = layers.Dense(p_dim * 2, activation='relu')(layer_final)
    layer_final = layers.Dense(1, activation='linear')(layer_final)
    model = keras.Model(inputs=input_sum_u,
                          outputs=layer_final, name="model")
    
    model.compile(optimizer='SGD', loss='mse')
    return embedding, model

# ====================================================== # 
#                                                        #
#       embedding                                        #
#                                                        #
# ====================================================== # 

def embed(embed_u_set, embed_x_set, embedding, g_link, T=T, p_dim=p_dim):
    n_nodes = len(g_link)
    for _ in range(T):
        embed_u_set_c = {}
        for node in embed_u_set.keys():
            embed_u_set_c[node] = embed_u_set[node].copy()

        tmp_cnt = {}; tmp_val = {}; 
        for node in range(n_nodes):
            cnt = 0
            val = 0
            for linked_node in g_link[node].keys():
                cnt += 1
                val += embed_x_set[linked_node]
                embed_u_set[node] += embed_u_set_c[linked_node]
            embed_u_set[node] -= embed_u_set_c[node]
            tmp_cnt[node] = cnt; tmp_val[node] = val; 
            
        x = np.array([np.array([tmp_cnt[i], tmp_val[i]]).reshape((1,2)) 
                      for i in range(n_nodes)]).reshape((n_nodes,2))
        u = np.array([embed_u_set[i].reshape((1,p_dim)) 
                      for i in range(n_nodes)]).reshape((n_nodes,p_dim))

        pred = embedding.predict([x, u])
        for node in range(n_nodes):
            embed_u_set[node] = pred[node]

    return embed_u_set            
    
def get_values(embed_u_set, embed_x_set, model, embedding, g_link): 
    n_nodes = len(g_link)
    embed_u_set = embed(embed_u_set, embed_x_set, embedding, g_link)
    embed_u_set_array =  np.array([embed_u_set[i] for i in range(n_nodes)])      
    sum_all_u = embed_u_set_array.sum(axis=0)
    shape_sum = np.array([sum_all_u for _ in range(n_nodes)])
    values = model.predict(np.concatenate((embed_u_set_array, shape_sum), axis = 1))
    return embed_u_set, embed_u_set_array, sum_all_u, values
    
# ====================================================== # 
#                                                        #
#       reinforcement learning                           #
#                                                        #
# ====================================================== #     

def stop(selected_set, edges):
    left_edges = edges.copy()
    for i in selected_set:
        left_edges = [j for j in left_edges if i not in j]
    return left_edges
    
def pick_action(cur_epi, selected_set, model, embed_u_set_array, sum_all_u):
    n_nodes = len(embed_u_set_array)
    available_actions = list(set(range(n_nodes)).difference(set(selected_set)))
    n_act = len(available_actions)
    if np.random.rand() < cur_epi:
        return random.choice(available_actions)
    else:
        tmp_embed =  np.array([embed_u_set_array[i] for i in available_actions])
        tmp_sum = np.array([sum_all_u for _ in range(n_act)])
        action_vals = model.predict(np.concatenate((tmp_embed, tmp_sum), axis = 1))
        return available_actions[np.argmax(action_vals.reshape(-1))]
    
def update_y(action, selected_set, embed_u_set, embed_x_set, 
             edges, init_len_state, model, embedding, g_link, k = 0, maxQ = 0):
    
    selected_set_c = selected_set.copy()
    selected_set_c.append(action)
    
    if k >= n_steps or stop(selected_set_c, edges) == []: 
        return - len(selected_set_c) + gamma * maxQ

    embed_u_set_c = {}
    for node in embed_u_set.keys():
        embed_u_set_c[node] = embed_u_set[node].copy()    
    embed_x_set_c = embed_x_set.copy()
    for i in selected_set:
        embed_x_set_c[i] = 1
    
    embed_u_set_c, embed_u_set_array_c, sum_all_u, values = \
    get_values(embed_u_set_c, embed_x_set_c, model, embedding, g_link)
    
    values = sorted([[values[i][0], i] for i in range(len(values))], reverse = True)
    #values = sorted([[values[i][0], i] for i in range(len(values))], )

    values = [i for i in values if i[1] not in selected_set_c]
    return max([update_y(i[1], selected_set_c, embed_u_set_c, embed_x_set_c, 
     edges, init_len_state, model, embedding, g_link, k = k+1, maxQ = i[0]) 
     for i in values])
    
    

# ====================================================== # 
#                                                        #
#       main                                             #
#                                                        #
# ====================================================== #     
  
embedding, model = build_model()

for figure in tqdm(range(50)):

    s1 = time.time()
    g, edges, g_link, n_nodes = generate_graph(show_pic = False) 
    s2 = time.time()
    per_memories = []
    see = []
    for replay in range(n_replay):
        
        s3 = time.time()
        embed_u_set = {i:np.ones(p_dim) * 0.01 for i in range(n_nodes)}
        embed_x_set = {i:0 for i in range(n_nodes)} 
        selected_set = []
        s4 = time.time()
        while stop(selected_set, edges) != []:
            
            if cur_epi * decay_epi < min_epi: pass
            else: cur_epi *= decay_epi

            s5 = time.time()
            embed_u_set, embed_u_set_array, sum_all_u, values = \
            get_values(embed_u_set, embed_x_set, model, embedding, g_link)
            
            s6 = time.time()
            action = pick_action(cur_epi, selected_set, model, embed_u_set_array, sum_all_u)
            
            Qvalues =  update_y(action, selected_set, embed_u_set, 
                                        embed_x_set, edges.copy(), len(selected_set), 
                                        model, embedding, g_link, k = 0, maxQ = 0) 
 
            see.append([selected_set.copy(), action, Qvalues])    
            s7 = time.time()
            embed_x_set[action] = 1 
            selected_set.append(action)          
            
            memories = \
            np.concatenate((embed_u_set_array[action].reshape((1,p_dim)),
            sum_all_u.reshape((1,p_dim))), axis = 1)
                
            if str(memories) not in per_memories:
                per_memories.append(str(memories))
                train_data.append([memories, Qvalues])
            s8 = time.time()
            s9 = time.time()
            #print(figure, round(s2-s1,2), round(s6-s5,2), round(s7-s6,2),)

    if len(train_data) > batch:
        batch_train = random.sample(train_data, batch)
        trx = list(zip(*batch_train))[0]
        trx = np.array([np.array(i[0]) for i in trx])
        model.fit(trx, np.array(list(zip(*batch_train))[1]).reshape(batch,1),
                  verbose = 0, epochs = 1000)


embedding.save('/Users/en-chengchang/Desktop/embedding')
model.save('/Users/en-chengchang/Desktop/model')


evaluation = []
for _ in range(5):
    s1 = time.time()
    g_val = nx.barabasi_albert_graph(15, 1, seed = _)
    edges_val = list(g_val.edges())
    g_link_val = g_val._adj
    n_nodes_val = len(g_link_val)
    nx.draw(g_val, with_labels=True, font_weight='bold')
    plt.show()
    
    
    embed_u_set = {i:np.ones(p_dim) * 0.01 for i in range(n_nodes_val)}
    embed_x_set = {i:0 for i in range(n_nodes_val)} 
    selected_set = []
    
    while stop(selected_set, edges_val) != []:
        
        embed_u_set, embed_u_set_array, sum_all_u, values = \
        get_values(embed_u_set, embed_x_set, model, embedding, g_link_val)
        
        
        action = pick_action(0, selected_set, model, embed_u_set_array, sum_all_u)

        embed_x_set[action] = 1 
        selected_set.append(action)   
    #evaluation.append([len(selected_set),selected_set[:3]])
    evaluation.append(selected_set)
   
