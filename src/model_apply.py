from os import path
import sys, json, time
import torch
import numpy as np
from model import Net
from dataset import NodeDataset

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Weights into network')

# Model Build output
model_path=path.join(BASE_DIR, 'data/model_build_outputs/mlp_state_dict.pt')

net = Net()
net.load_state_dict(torch.load(model_path))

print('Reading new dataset')
test_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
test_travel_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
test_pacakge_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_package_data.json')

dataset = NodeDataset(test_pacakge_path, test_travel_path, test_routes_path, train=False)

routes = {route_id: {'proposed': {}} for route_id in data_set.route_ids}

def route_sequencer(start_node, travel_time, start_time, end_time, volume, service):
    #Parameters
    available_nodes = set(range(travel_time.shape[0]))
    visited_nodes = set([])
    sequence = []
    visited_nodes.add(start_node)
    sequence.append(start_node)
    current = start_node
    
    #Initial state
    d = distance[start_node, :].view(-1,1)
    
    x = torch.cat([d, service, volume, start_time, end_time], dim=1)
    
    for i in range(travel_time.shape[0]):
        if len(visited_nodes) == travel_time.shape[0]:
            break
        
        idx = list(available_nodes-visited_nodes)
        evals = net(x[idx,:]).view(1,-1)
        selected_node = idx[torch.argmax(evals)]
        
        visited_nodes.add(selected_node)
        sequence.append(selected_node)
        current = selected_node
        
        #Create new x
        d = distance[current, :].view(-1,1)
        volume[selected_node] = 0
        service[selected_node] = 0
        x = torch.cat([d, service, volume, start_time, end_time], dim=1)
        
    return sequence

for route_id in tqdm(data_set.route_ids, total=len(data_set.route_ids)):
    keys = np.array(list(data_set.routes[route_id]['stops'].keys()))
    distance = data_set.distance_matrix(route_id)
    ser, vol, st, et = data_set.parse_packages(route_id)
    start_node = data_set.get_start_node(route_id)
    ser, vol, st, et = ser.view(-1,1), vol.view(-1,1), st.view(-1,1), et.view(-1,1)
    
    sequence = route_sequencer(start_node, distance, st, et, vol, ser)
    
    for i, stop in enumerate(keys[sequence]):
        routes[route_id]['proposed'][stop] = i

# Write output data
output_path = path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')

with open(output_path, 'w') as out_file:
    json.dump(routes, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')
