import torch
import torch
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from tqdm import tqdm
import pickle as pkl
import numpy as np

class Cb513(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['cb513_nr.pkl', 'cb513_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['cb513_q8_nr.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        seq_length, adjacency_matrices = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adjacency_matrices.append(np.vstack((data[pdbid][-1].row, data[pdbid][-1].col)))
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )

        # adjacency_matrices = torch.tensor(adjacency_matrices)
        # seq_encodings = torch.tensor(seq_encodings)

        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            y = torch.tensor(seq_encodings[i]['labels'], dtype=torch.long)
            edge_index = torch.tensor(adjacency_matrices[i], dtype=torch.long)
            data = Data(y=y, edge_index=edge_index, seq_encodings=seq_encodings[i])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
# p = Cb513()
# print(p[0])
class Casp(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['casp12_nr.pkl', 'casp12_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['casp12_q8_nr.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        seq_length, adjacency_matrices = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adjacency_matrices.append(np.vstack((data[pdbid][-1].row, data[pdbid][-1].col)))
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )

        # adjacency_matrices = torch.tensor(adjacency_matrices)
        # seq_encodings = torch.tensor(seq_encodings)

        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            y = torch.tensor(seq_encodings[i]['labels'], dtype=torch.long)
            edge_index = torch.tensor(adjacency_matrices[i], dtype=torch.long)
            data = Data(y=y, edge_index=edge_index, seq_encodings=seq_encodings[i])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Ts115(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational/',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['ts115_nr.pkl', 'ts115_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['ts115_q8_nr.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        seq_length, adjacency_matrices = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adjacency_matrices.append(np.vstack((data[pdbid][-1].row, data[pdbid][-1].col)))
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )

        # adjacency_matrices = torch.tensor(adjacency_matrices)
        # seq_encodings = torch.tensor(seq_encodings)

        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            y = torch.tensor(seq_encodings[i]['labels'], dtype=torch.long)
            edge_index = torch.tensor(adjacency_matrices[i], dtype=torch.long)
            data = Data(y=y, edge_index=edge_index, seq_encodings=seq_encodings[i])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ProteinDataset(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['df_final_nr.pkl', 'train_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['df_final_q8_nr.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        seq_length, adjacency_matrices = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adjacency_matrices.append(np.vstack((data[pdbid][-1].row, data[pdbid][-1].col)))
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )
        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            y = torch.tensor(seq_encodings[i]['labels'], dtype=torch.long)
            edge_index = torch.tensor(adjacency_matrices[i], dtype=torch.long)
            data = Data(y=y, edge_index=edge_index, seq_encodings=seq_encodings[i])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ProteinDatasetForRGCN(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['df_final_q8_relational.pkl', 'train_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['df_final_q8_relational.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        adj_mats, edge_types = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            adj_matrix_head, adj_matrix_tail, edge_type = [], [], []
            for j in range(3):
                adj_matrix_head.extend(data[pdbid][j].nonzero()[0])
                adj_matrix_tail.extend(data[pdbid][j].nonzero()[1])
                edge_type.extend([j]*len(data[pdbid][j].nonzero()[0]))
            adj_matrix_head = torch.tensor(adj_matrix_head)
            adj_matrix_tail = torch.tensor(adj_matrix_tail)
            edge_type = torch.tensor(edge_type)
            adj_mats.append(torch.cat((adj_matrix_head.unsqueeze(0), adj_matrix_tail.unsqueeze(0)), dim=0))
            edge_types.append(edge_type)
            # seq_encodings.append(
            #     {
            #         "input_ids": torch.tensor(language_data.input_ids[i]),
            #         "attention_mask": torch.tensor(language_data.attention_mask[i]),
            #         "token_type_ids": torch.tensor(language_data.token_type_ids[i]),
            #         "labels": torch.tensor(language_data.labels[i])
            #     }
            # )
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )
        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            data = Data(
                edge_index=adj_mats[i],
                edge_type=edge_types[i],
                seq_encodings=seq_encodings[i]
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class RelationalCaspDataset(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['casp12_final_q8_relational.pkl', 'casp12_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['casp12_q8_relational.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        adj_mats, edge_types = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adj_matrix_head, adj_matrix_tail, edge_type = [], [], []
            for j in range(3):
                adj_matrix_head.extend(data[pdbid][j].nonzero()[0])
                adj_matrix_tail.extend(data[pdbid][j].nonzero()[1])
                edge_type.extend([j]*len(data[pdbid][j].nonzero()[0]))
            adj_matrix_head = torch.tensor(adj_matrix_head)
            adj_matrix_tail = torch.tensor(adj_matrix_tail)
            edge_type = torch.tensor(edge_type)
            adj_mats.append(torch.cat((adj_matrix_head.unsqueeze(0), adj_matrix_tail.unsqueeze(0)), dim=0))
            edge_types.append(edge_type)
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )
        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            # data = Data(y=y, edge_index=edge_index, seq_len=seq_length[i], seq_encodings=seq_encodings[i])
            data = Data(
                edge_index=adj_mats[i],
                edge_type=edge_types[i],
                seq_encodings=seq_encodings[i]
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class RelationalCb513Dataset(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['cb513_final_q8_relational.pkl', 'cb513_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['cb513_q8_relational.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        adj_mats, edge_types = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adj_matrix_head, adj_matrix_tail, edge_type = [], [], []
            for j in range(3):
                adj_matrix_head.extend(data[pdbid][j].nonzero()[0])
                adj_matrix_tail.extend(data[pdbid][j].nonzero()[1])
                edge_type.extend([j]*len(data[pdbid][j].nonzero()[0]))
            adj_matrix_head = torch.tensor(adj_matrix_head)
            adj_matrix_tail = torch.tensor(adj_matrix_tail)
            edge_type = torch.tensor(edge_type)
            adj_mats.append(torch.cat((adj_matrix_head.unsqueeze(0), adj_matrix_tail.unsqueeze(0)), dim=0))
            edge_types.append(edge_type)
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )
        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            # data = Data(y=y, edge_index=edge_index, seq_len=seq_length[i], seq_encodings=seq_encodings[i])
            data = Data(
                edge_index=adj_mats[i],
                edge_type=edge_types[i],
                seq_encodings=seq_encodings[i]
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class RelationalTs115Dataset(InMemoryDataset):
    def __init__(self,
                 root='/Data/deeksha/pssp/ProtTrans/data/q8_data/relational',
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # creat two variable to store the trained data and the pdbids
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['ts115_final_q8_relational.pkl', 'ts115_tokenized_dataset.pkl']
    
    @property
    def processed_file_names(self):
        return ['ts115_q8_relational.pt']
    
    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            data = pkl.load(f)
        with open(self.raw_paths[1], 'rb') as f:
            language_data = pkl.load(f)

        adj_mats, edge_types = [], []
        seq_encodings = []
        for i, pdbid in enumerate(tqdm(language_data.keys())):
            if pdbid not in data.keys():
                continue
            adj_matrix_head, adj_matrix_tail, edge_type = [], [], []
            for j in range(3):
                adj_matrix_head.extend(data[pdbid][j].nonzero()[0])
                adj_matrix_tail.extend(data[pdbid][j].nonzero()[1])
                edge_type.extend([j]*len(data[pdbid][j].nonzero()[0]))
            adj_matrix_head = torch.tensor(adj_matrix_head)
            adj_matrix_tail = torch.tensor(adj_matrix_tail)
            edge_type = torch.tensor(edge_type)
            adj_mats.append(torch.cat((adj_matrix_head.unsqueeze(0), adj_matrix_tail.unsqueeze(0)), dim=0))
            edge_types.append(edge_type)
            seq_encodings.append(
                {
                    "input_ids": torch.tensor(language_data[pdbid]['input_ids']),
                    "attention_mask": torch.tensor(language_data[pdbid]['attention_mask']),
                    "token_type_ids": torch.tensor(language_data[pdbid]['token_type_ids']),
                    "labels": torch.tensor(language_data[pdbid]['labels'])
                }
            )
        data_list = []
        for i in tqdm(range(len(seq_encodings))):
            # data = Data(y=y, edge_index=edge_index, seq_len=seq_length[i], seq_encodings=seq_encodings[i])
            data = Data(
                edge_index=adj_mats[i],
                edge_type=edge_types[i],
                seq_encodings=seq_encodings[i]
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
