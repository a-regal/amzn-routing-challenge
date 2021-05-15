from os import path
import sys, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import NodeDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import Net

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Input Data')
training_routes_path = path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
training_travel_path = path.join(BASE_DIR, 'data/model_build_inputs/travel_times.json')
training_pacakge_path = path.join(BASE_DIR, 'data/model_build_inputs/package_data.json')
training_sequence_path = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
dataset = NodeDataset(training_pacakge_path, training_travel_path, training_routes_path, training_sequence_path)

train_loader = DataLoader(dataset, batch_size=1)

net = Net()
criterion = nn.SmoothL1Loss()
opt = optim.Adam(net.parameters(),1e-3)

print('Begin training')
for i in tqdm(range(500)):
    for i, data in enumerate(train_loader, 0):
        x, y = data
        x, y = x.squeeze(), y.view(-1,1)
        out = net(x)
        opt.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        opt.step()

print('Save model')
model_path=path.join(BASE_DIR, 'data/model_build_outputs/mlp_state_dict.pt')
torch.save(net.state_dict(), model_path)