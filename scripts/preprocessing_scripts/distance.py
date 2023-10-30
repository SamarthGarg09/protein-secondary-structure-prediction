import os
import pickle
import numpy as np
import re
from Bio import PDB
from tqdm.auto import tqdm
parser = PDB.PDBParser()
io = PDB.PDBIO()


count, error_ids = 0, []

for (root,dirs,files) in os.walk('/Data/deeksha/disha/ProtTrans/data/pdbid_files'):
    dic_seq = {}
    progress_bar = tqdm(files, desc='pdb_files', leave=False)
    for k, filename in enumerate(files):
        print(k, filename)
        # f = open(os.path.join(root,filename),'r').read().split('\n')
        id_ = filename.split('.')[0]
        d = {}
        try:
            struct = parser.get_structure(id_,os.path.join(root,filename))
            for model in struct:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            x,y,z = atom.get_coord()
                            try:
                                d[residue.get_resname()].append([x,y,z])
                            except:
                                d[residue.get_resname()] = [x,y,z]
        except UnicodeDecodeError as exp:    
            count += 1
            error_ids.append(id_)
            continue
        progress_bar.update(1)

                        # print(x,y,z)
        dic_seq[id_] = d
        # print(dic_seq)
        
    print(f'No. of errors encounteres : {count}')

    with open("/Data/deeksha/disha/ProtTrans/data/error_ids.txt", 'w') as f:
        f.write(",".join(error_ids))

    pickle.dump(dic_seq, open('dic_seq.pkl','wb'))
