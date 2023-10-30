# %%
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
import logging


# %%
# logging.basicConfig(level=logging.INFO, filename='/Data/deeksha/disha/ProtTrans/scripts/train/q3-5epochs/metrics.log', filemode='w')

tokenizer_name = 'yarongef/DistilProtBert'
model_name = tokenizer_name
# model_name = "yarongef/DistilProtBert"
max_length = 1024
print("Loading Tokenizer")

# %%
def load_dataset(path, max_length):
        df = pd.read_excel(path)
        df.rename(columns={'input_x':'input', 'input_y': 'npz'}, inplace=True)
        print(df.columns)
        df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
        df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
        seqs = [ list(seq)[:max_length-2] for seq in df['input_fixed']]

        df['label_fixed'] = ["".join(label.split()) for label in df[' dssp3']]
        labels = [ list(label)[:max_length-2] for label in df['label_fixed']]

        df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df[' disorder']]
        disorder = [ disorder.split()[:max_length-2] for disorder in df['disorder_fixed']]

        assert len(seqs) == len(labels) == len(disorder)
        return seqs, labels, disorder, df['pdbid'].tolist()

train_seqs, train_labels, train_disorder, train_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/df_train.xlsx', max_length)
val_seqs, val_labels, val_disorder, val_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/df_test.xlsx', max_length)
casp12_test_seqs, casp12_test_labels, casp12_test_disorder, casp_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/casp12_final.xlsx', max_length)
cb513_test_seqs, cb513_test_labels, cb513_test_disorder, cb513_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/cb513_final.xlsx', max_length)
ts115_test_seqs, ts115_test_labels, ts115_test_disorder, ts115_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/ts115_final.xlsx', max_length)
     
print(train_seqs[0][10:30], train_labels[0][10:30], train_disorder[0][10:30], sep='\n')
     
seq_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, do_lower_case=False)
  
train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
casp12_test_seqs_encodings = seq_tokenizer(casp12_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
cb513_test_seqs_encodings = seq_tokenizer(cb513_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
ts115_test_seqs_encodings = seq_tokenizer(ts115_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

unique_tags = set(tag for doc in train_labels for tag in doc)
unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels_encodings = encode_tags(train_labels, train_seqs_encodings)
val_labels_encodings = encode_tags(val_labels, val_seqs_encodings)
casp12_test_labels_encodings = encode_tags(casp12_test_labels, casp12_test_seqs_encodings)
cb513_test_labels_encodings = encode_tags(cb513_test_labels, cb513_test_seqs_encodings)
ts115_test_labels_encodings = encode_tags(ts115_test_labels, ts115_test_seqs_encodings)


# %%


# %%
class SS3Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
# _ = train_seqs_encodings.pop("offset_mapping")
# _ = val_seqs_encodings.pop("offset_mapping")
# _ = casp12_test_seqs_encodings.pop("offset_mapping")
# _ = cb513_test_seqs_encodings.pop("offset_mapping")
# _ = ts115_test_seqs_encodings.pop("offset_mapping")

train_dataset = SS3Dataset(train_seqs_encodings, train_labels_encodings)
val_dataset = SS3Dataset(val_seqs_encodings, val_labels_encodings)
casp12_test_dataset = SS3Dataset(casp12_test_seqs_encodings, casp12_test_labels_encodings)
cb513_test_dataset = SS3Dataset(cb513_test_seqs_encodings, cb513_test_labels_encodings)
ts115_test_dataset = SS3Dataset(ts115_test_seqs_encodings, ts115_test_labels_encodings)

# %%
import pickle
with open("/Data/deeksha/disha/ProtTrans/data/multimodel_data_dir/raw/casp12_test_dataset.pkl", 'rb') as f:
    casp12_test_dataset = pickle.load(f)

# %%
with open("/Data/deeksha/disha/ProtTrans/data/multimodel_data_dir/raw/tokenized_df_final_q3.pkl", 'rb') as f2:
    t = pickle.load(f2)


# %%
# ssave all the test datasets in the pickle file
import pickle
with open('/Data/deeksha/disha/ProtTrans/data/multimodel_data_dir/raw/casp12_test_dataset.pkl', 'wb') as f:
    pickle.dump(casp12_test_dataset, f)
with open('/Data/deeksha/disha/ProtTrans/data/multimodel_data_dir/raw/cb513_test_dataset.pkl', 'wb') as f:
    pickle.dump(cb513_test_dataset, f)
with open('/Data/deeksha/disha/ProtTrans/data/multimodel_data_dir/raw/ts115_test_dataset.pkl', 'wb') as f:
    pickle.dump(ts115_test_dataset, f)


# %%
import pickle as pkl

with open("/Data/deeksha/disha/ProtTrans/data/multimodel_data_dir/raw/tokenized_df_final_q3.pkl", 'rb') as f:
    data = pkl.load(f)

# %%
data.keys()

# %%
torch.allclose(torch.tensor(data[1000].ids),train_dataset[1000]['input_ids'])

# %%
len(data['pdbid']), len(train_dataset), len(val_dataset)

# %%
train_dataset[0]

# %% [markdown]
# #### Above calculation ensures that tokenized df_file is pickled correctly

# %%



