import json
import torch
import numpy as np
from torch.utils.data import Dataset

class NodeDataset(Dataset):
    def __init__(self, package_file_path, stop_file_path, route_file_path, sequence_file_path=None, train=True):
        #Read JSON files. This is slow, might need to look for a faster alternative
        with open(package_file_path, 'r') as f:
            self.packages = json.load(f)

        with open(stop_file_path, 'r') as f:
            self.travel_times = json.load(f)

        with open(route_file_path, 'r') as f:
            self.routes = json.load(f)

        #Store a list with the total route IDs for the parsing in getitem
        if train:
            with open(sequence_file_path, 'r') as f:
                self.sequences = json.load(f)
            self.route_ids = [key for key in self.routes.keys() if self.routes[key]['route_score'] == 'High']
            self.train = True
        else:
            self.route_ids = [key for key in self.routes.keys()]
            self.train = False

    def __len__(self):
        return len(self.route_ids)

    def __getitem__(self, idx):
        route_id = self.route_ids[idx]
        distance = self.distance_matrix(route_id)
        ser, vol, st, et = self.parse_packages(route_id)
        start_node = self.get_start_node(route_id)
        ser, vol, st, et = ser.view(-1,1), vol.view(-1,1), st.view(-1,1), et.view(-1,1)


        if self.train:
            d = torch.zeros_like(st)
            seq = self.get_sequence(route_id)
            ser, vol, st, et = ser[seq], vol[seq], st[seq], et[seq]
            
            counter = 0
            for i, j in zip(seq, seq[1:]):
                d[counter] = distance[i,j]
            
            x = torch.cat([d,ser,vol,st,et], dim=1)
            y = 1. / torch.arange(1,vol.shape[0]+1)
            
            return x, y
        
        else:
            d = distance[start_node, :]
            return torch.cat([d.view(-1,1),ser,vol,st,et], dim=1)
            

    def distance_matrix(self, route_id):
        mat = np.zeros((len(self.routes[route_id]['stops']), len(self.routes[route_id]['stops'])))

        for i, stop in enumerate(self.travel_times[route_id]):
            mat[i, :] = list(self.travel_times[route_id][stop].values())

        return torch.tensor(mat, dtype=torch.float32)

    def get_sequence(self,route_id):
        actual = self.sequences[route_id]['actual']
        d = dict(zip(range(len(actual)), actual.values()))
        a = list(dict(sorted(d.items(), key=lambda item: item[1])).keys())
        return torch.tensor(a)

    def get_start_node(self,route_id):
        start_node = -1
        for i, idx in enumerate(self.routes[route_id]['stops']):
            if self.routes[route_id]['stops'][idx]['type'] != 'Dropoff':
                start_node = i

        return start_node

    def parse_packages(self, route_id):
        service_time = []
        volume = []
        start_times = []
        end_times = []

        for stop in self.packages[route_id]:
            total_service_time = 0
            total_volume = 0
            s, e = 0, 0
            for package in self.packages[route_id][stop]:
                start_time = self.packages[route_id][stop][package]['time_window']['start_time_utc']
                end_time = self.packages[route_id][stop][package]['time_window']['end_time_utc']
                if type(start_time) == str:
                    s, e = start_time[-8:], end_time[-8:]
                    s = int(s[:2])*60 + int(s[3:5])
                    e = int(e[:2])*60 + int(e[3:5])
                else:
                    s, e = 0, 0

                #Add total service time
                total_service_time += self.packages[route_id][stop][package]['planned_service_time_seconds']

                #Add total volume
                total_volume += np.prod(list(self.packages[route_id][stop][package]['dimensions'].values()))

            service_time.append(total_service_time / 60)
            volume.append(total_volume)
            start_times.append(s)
            end_times.append(e)

        return torch.tensor(service_time, dtype=torch.float32), torch.tensor(volume, dtype=torch.float32), torch.tensor(start_times, dtype=torch.float32), torch.tensor(end_times, dtype=torch.float32)