import pickle as pkl
from transformers import BertTokenizerFast
import pandas as pd
import numpy as np
import re

# logging.basicConfig(level=logging.INFO, filename='/Data/deeksha/disha/ProtTrans/scripts/train/q3-5epochs/metrics.log', filemode='w')

tokenizer_name = 'yarongef/DistilProtBert'
model_name = tokenizer_name
# model_name = "yarongef/DistilProtBert"
max_length = 1024
print("Loading Tokenizer")

def load_dataset(path, max_length):
        df = pd.read_excel(path)
        df.rename(columns={'input_x':'input', 'input_y': 'npz'}, inplace=True)
        print(df.columns)
        df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
        df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
        seqs = [ list(seq)[:max_length-2] for seq in df['input_fixed']]

        df['label_fixed'] = ["".join(label.split()) for label in df['dssp8']]
        labels = [ list(label)[:max_length-2] for label in df['label_fixed']]

        df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df[' disorder']]
        disorder = [ disorder.split()[:max_length-2] for disorder in df['disorder_fixed']]

        assert len(seqs) == len(labels) == len(disorder)
        return seqs, labels, disorder, df['pdbid'].tolist()

train_seqs, train_labels, train_disorder, train_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/df_final.xlsx', max_length)
# val_seqs, val_labels, val_disorder, val_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/df_test.xlsx', max_length)

casp12_test_seqs, casp12_test_labels, casp12_test_disorder, casp_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/casp12_final.xlsx', max_length)
cb513_test_seqs, cb513_test_labels, cb513_test_disorder, cb513_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/cb513_final.xlsx', max_length)
ts115_test_seqs, ts115_test_labels, ts115_test_disorder, ts115_ids = load_dataset('/Data/deeksha/disha/ProtTrans/data/final_excel_files/ts115_final.xlsx', max_length)
     
print(train_seqs[0][10:30], train_labels[0][10:30], train_disorder[0][10:30], sep='\n')
     
seq_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, do_lower_case=False)
  
train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
# val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
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
# val_labels_encodings = encode_tags(val_labels, val_seqs_encodings)
casp12_test_labels_encodings = encode_tags(casp12_test_labels, casp12_test_seqs_encodings)
cb513_test_labels_encodings = encode_tags(cb513_test_labels, cb513_test_seqs_encodings)
ts115_test_labels_encodings = encode_tags(ts115_test_labels, ts115_test_seqs_encodings)

# key value pair where key is pdbid and value is a list of input_ids, attention_mask, labels, token_type_ids

di_train = {}
for i in range(len(train_ids)):
    di_train[train_ids[i]] = {
        "input_ids": train_seqs_encodings["input_ids"][i],
        "attention_mask": train_seqs_encodings["attention_mask"][i],
        "labels": train_labels_encodings[i],
        "token_type_ids": train_seqs_encodings["token_type_ids"][i]
    }

with open('/Data/deeksha/disha/ProtTrans/data/q8_data/relational/raw/train_dataset.pkl', 'wb') as f:
    pkl.dump(di_train, f)

di = {}
for i in range(len(casp_ids)):
    di[casp_ids[i]] = {
        "input_ids": casp12_test_seqs_encodings["input_ids"][i],
        "attention_mask": casp12_test_seqs_encodings["attention_mask"][i],
        "labels": casp12_test_labels_encodings[i],
        "token_type_ids": casp12_test_seqs_encodings["token_type_ids"][i]
    }

with open('/Data/deeksha/disha/ProtTrans/data/q8_data/relational/raw/casp12_test_dataset.pkl', 'wb') as f:
    pkl.dump(di, f)

di_cb513 = {}
for i in range(len(cb513_ids)):
    di_cb513[cb513_ids[i]] = {
        "input_ids": cb513_test_seqs_encodings["input_ids"][i],
        "attention_mask": cb513_test_seqs_encodings["attention_mask"][i],
        "labels": cb513_test_labels_encodings[i],
        "token_type_ids": cb513_test_seqs_encodings["token_type_ids"][i]
    }

with open('/Data/deeksha/disha/ProtTrans/data/q8_data/relational/raw/cb513_test_dataset.pkl', 'wb') as f:
    pkl.dump(di_cb513, f)

di_ts115 = {}
for i in range(len(ts115_ids)):
    di_ts115[ts115_ids[i]] = {
        "input_ids": ts115_test_seqs_encodings["input_ids"][i],
        "attention_mask": ts115_test_seqs_encodings["attention_mask"][i],
        "labels": ts115_test_labels_encodings[i],
        "token_type_ids": ts115_test_seqs_encodings["token_type_ids"][i]
    }

with open('/Data/deeksha/disha/ProtTrans/data/q8_data/relational/raw/ts115_test_dataset.pkl', 'wb') as f:
    pkl.dump(di_ts115, f)

print("Done")