#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 17:27:04 2026

@author: zyuan
"""

import torch
import matplotlib.pyplot as plt
cp = torch.load('checkpoint_epoch_1.pt')
traj = cp['rollout_trajectories'] 

traj = traj.detach().cpu().numpy()
plt.figure()
for i in range(len(traj)):
    plt.plot(traj[i,:,0], traj[i,:,1])
    plt.show()


