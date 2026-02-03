#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:18:47 2026

@author: zyuan
"""

from datasets import load_dataset
import json
# Load dataset using HuggingFace datasets
ds = load_dataset("NathanGavenski/LunarLander-v2")

train = ds['train']
start_indices = []
for i in range(len(train)):
    if train[i]['episode_starts'] == 1:
        start_indices.append(i)

cnt = 0
Trajectories = {}
for i in range(len(start_indices)-1):
    trajectory = train[start_indices[i]:start_indices[i+1]]
    traj = {'Traj_index': cnt,
            'Trajectory': trajectory
            }
    Trajectories[cnt] = traj


with open('LunarLander.json', 'w', encoding='utf-8') as f:
    json.dump(Trajectories, f, ensure_ascii=False, indent=4)
