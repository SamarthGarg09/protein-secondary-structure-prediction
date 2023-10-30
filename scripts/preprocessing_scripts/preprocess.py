import numpy as np

# load npz file
casp12 = np.load('/Data/deeksha/disha/ProtTrans/data/CASP12_HHblits.npz', allow_pickle=True)
cb513 = np.load("/Data/deeksha/disha/ProtTrans/data/CB513_HHblits.npz", allow_pickle=True)
train = np.load("/Data/deeksha/disha/ProtTrans/data/Train_HHblits.npz", allow_pickle=True)
ts115 = np.load("/Data/deeksha/disha/ProtTrans/data/TS115_HHblits.npz", allow_pickle=True)

# extract data
casp12_data = casp12['data']
cb513_data = cb513['data']
train_data = train['data']
ts115_data = ts115['data']

casp12_pdb = casp12['pdbids']
cb513_pdb = cb513['pdbids']
train_pdb = train['pdbids']
ts115_pdb = ts115['pdbids']

print(casp12_data.shape)
print(casp12_pdb.shape)
print(cb513_data.shape)
print(cb513_pdb.shape)
print(train_data.shape)
print(train_pdb.shape)
print(ts115_data.shape)
print(ts115_pdb.shape)

import requests
from tqdm.auto import tqdm
import os

pdb_folder_name='pdbfiles_alt'
pdb_list=[casp12_pdb,cb513_pdb,train_pdb,ts115_pdb]


def fetch_pdb_files():
    base_url = 'https://files.rcsb.org/download/'
    c=0
    for entry in tqdm(pdb_chain_list, desc='Fetching PDB files', leave=False):
        pdb_id, chain_id = entry.split('-')
        pdb_id = pdb_id.upper()
        file_name = pdb_id + '_' + chain_id.upper() + '.pdb'
        if os.path.exists(f'../data/{pdb_folder_name}/'+file_name):
            c += 1
            print(c)
            continue     
        else:
            try:
                os.mkdir(f'../data/{pdb_folder_name}')
            except FileExistsError:
                print('Folder exists')
        url = base_url + pdb_id + '.pdb'
        response = requests.get(url)
        
        if response.status_code == 200:
            file_name = pdb_id + '_' + chain_id.upper() + '.pdb'
            
            with open(f'../data/{pdb_folder_name}/'+file_name, 'w') as file:
                # only write if file does not exisis
                if not os.path.exists(f'../data/{pdb_folder_name}/'+file_name):
                    file.write(response.text)
            
            print(f'Successfully fetched {file_name}')
        else:
            print(f'Failed to fetch PDB file for {pdb_id}-{chain_id}')

for file in pdb_list:
    # fetch_pdb_files(casp12_pdb)
    # fetch_pdb_files(cb513_pdb)
    # fetch_pdb_files(train_pdb)
    fetch_pdb_files(file)