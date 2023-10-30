import numpy as np
import pickle
import os
import pickle
import numpy as np
import re
from Bio import PDB
import time
from tqdm.auto import tqdm
import scipy.sparse as sp

parser = PDB.PDBParser()
io = PDB.PDBIO()

print('loading pickle file')

t1 = time.time()
dic_seq = pickle.load(open('/Data/deeksha/disha/ProtTrans/data/dic_seq.pkl','rb'))
t2 = time.time()
print('time taken to load pickle file', t2-t1)

amino_acid_dic = {'G':'Gly','P':'Pro','A':'Ala','V':'Val','L':'Leu','I':'Ile','M':'Met','C':'Cys','F':'Phe','Y':'Tyr','W':'Trp','H':'His','K':'Lys','R':'Arg','Q':'Gln','N':'Asn','E':'Glu','D':'Asp','S':'Ser','T':'Thr','X':'Xaa'}

def get_points(id_, seq):
    distances = dic_seq[id_]
    pts = []
    for m,amino in enumerate(seq):
        amino_acid = amino_acid_dic[amino]
        try:
            pts.append(distances[amino_acid.upper()][-1])
        except:
            pts.append(distances['ALA'][-1])
    return pts

def get_distances(pts):
    x = np.array([float(pt[0]) for pt in pts])
    y = np.array([float(pt[1]) for pt in pts])
    z = np.array([float(pt[2]) for pt in pts])
    M = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)) + np.square(z - z.reshape(-1,1)))
    M[M != 0] = 1 
    return M

f = open('/Data/deeksha/disha/ProtTrans/data/q8_data/text_file/netsurf.txt','r')
x = f.read().split('\n')

max_len = []

for i in range(len(x)):
    if '>' in x[i]:
        max_len.append(len(x[i+1]))

def one_hot_encode_primary(seq):
    seq = re.sub(r"[UZOB]", "X", seq)
    mapping = dict(zip("ACDEFGHIKLMNPQRSTVWYX", range(21)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(21)[seq2]

def one_hot_encode_secondary(seq):
    mapping = dict(zip("GHIBESTC", range(8)))
    # mapping = dict(zip("HEC", range(3)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(8)[seq2]
    # return np.eye(3)[seq2]
#------------------------------------------------------------



data_X = np.zeros((len(max_len), 700, 21),dtype=float)
data_Y = np.zeros((len(max_len), 700, 8),dtype=float)
# data_Y = np.zeros((len(max_len), 700, 3),dtype=float)

c = 0
data_dic = {}
list_ids = []
list_l = []

Z = np.zeros(21)
progress_bar = tqdm(total=len(max_len))
for i,line in enumerate(x):
    if '>' in line:
        id_ =  x[i][1:]
        if len(x[i+1]) > 700:
            seq = x[i+1] = x[i+1][:700]
            # x[i+2] = x[i+2][:700]
            x[i+3] = x[i+3][:700]
            # data_XX = np.ones((len(x[i + 2]), 1),dtype=float) 
            data_XX = np.ones((len(x[i + 3]), 1),dtype=float)
            try:
                pts = get_points(id_, seq)
                M = get_distances(pts)
            except:
                M = np.zeros((700, 700), dtype=float)
            M = sp.coo_matrix(M)
            # data_dic[id_] = [one_hot_encode_primary(x[i+1]), one_hot_encode_secondary(x[i+2]) , M] 
            data_dic[id_] = [one_hot_encode_primary(x[i+1]), one_hot_encode_secondary(x[i+3]) ,max_len[c], M] 

        else:
            seq = x[i+1]
            # data_XX = np.ones((len(x[i + 2]), 1),dtype=float)
            # data_XX = np.ones((len(x[i + 3]), 1),dtype=float)
            try:
                pts = get_points(id_, seq)
                M = get_distances(pts)
                L = np.zeros((700, 700), dtype=float)
                L[0: len(M), 0: len(M)] = M
            except:
                L = np.zeros((700, 700), dtype=float)
            L = sp.coo_matrix(L)
            data_dic[id_] = [np.append(one_hot_encode_primary(x[i+1]), np.zeros((700 - len(x[i+1]) , 21)) , axis = 0), np.append(one_hot_encode_secondary(x[i+2]) , np.zeros((700 - len(x[i+2]) , 8)) , axis = 0), L] 
            # data_dic[id_] = [np.append(one_hot_encode_primary(x[i+1]), np.zeros((700 - len(x[i+1]) , 21)) , axis = 0), np.append(one_hot_encode_secondary(x[i+3]) , np.zeros((700 - len(x[i+2]) , 3)) , axis = 0), max_len[c], L] 

        c += 1
        progress_bar.update(1)

print('No of samples: ',c)


train_dic = dict(list(data_dic.items()))

print('kk',len(train_dic))

pickle.dump(train_dic, open('/Data/deeksha/disha/ProtTrans/data/q8_data/relational/raw/df_final_nr.pkl','wb'))
t3 = time.time()
print('time taken to dump pickle file', t3-t1)