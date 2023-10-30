import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import re
import random
import evaluate
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence

metric = evaluate.load("seqeval")

def load_dataset(path):
    df = pd.read_excel(path)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    return df

df = load_dataset("/Data/deeksha/pssp/ProtTrans/data/final_excel_files/df_final.xlsx")

df['input_x'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_x']]
X = df['input_x'].values
y = df[' dssp3'].values


vocab = list(set([i for i in X for i in i.split()]))
vocab_size = len(vocab)

# create char to int mapping
char2int = dict((c, i) for i, c in enumerate(vocab))
int2char = dict((i, c) for i, c in enumerate(vocab))

label2int = {'PAD':0, 'C': 1, 'E': 2, 'H': 3}
int2label = {v:k for k, v in label2int.items()}

# label2int = {'PAD': 0, 'B': 1, 'C': 2, 'E': 3, 'G': 4, 'H': 5, 'I': 6, 'S': 7, 'T': 8}
# int2label = {0: 'PAD', 1: 'B', 2: 'C', 3: 'E', 4: 'G', 5: 'H', 6: 'I', 7: 'S', 8: 'T'}

def encode_sequence(seq, mapping):
    seq = seq.split()
    
    seq = [mapping[i] for i in seq]
    return seq
# Convert data into input and target tensors

def custom_collate_fn(batch):
    # Unzip the batch to separate the sequences and labels
    sequences, labels = zip(*batch)
    
    # Determine the maximum sequence length in this batch
    max_seq_length = max([len(seq) for seq in sequences])
    
    # Pad sequences and labels to max_seq_length
    padded_sequences = [seq + [0]*(max_seq_length - len(seq)) for seq in sequences]
    padded_labels = [lbl + [0]*(max_seq_length - len(lbl)) for lbl in labels]
    
    # Convert padded sequences and labels to tensors
    sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
    
    return sequences_tensor, labels_tensor

X = [encode_sequence(sample, char2int) for sample in X]
Y = [encode_sequence(sample, label2int) for sample in y]

train_size = int(0.8*len(X))
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

train_ds = list(zip(X_train, Y_train))
test_ds = list(zip(X_test, Y_test))

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)

class CNNBiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, num_filters, hidden_dim, num_layers):
        super(CNNBiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.bigru = nn.GRU(num_filters, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, num_labels)

    def forward(self, x):
        # Embedding Layer
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        # Conv Layer
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # BiGRU Layer
        x, _ = self.bigru(x)
        # Fully connected layer
        x = self.fc(x)
        return x
    
class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, num_filters, hidden_dim, num_layers):
        super(CNNBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(num_filters, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, num_labels)

    def forward(self, x):
        # Embedding Layer
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        # Conv Layer
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # BiLSTM Layer
        x, _ = self.bilstm(x)
        # Fully connected layer
        x = self.fc(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNNBiLSTM(len(char2int), 64, len(label2int), 64, 128, 2)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Prepare test dataset
cb513_df = load_dataset("/Data/deeksha/pssp/ProtTrans/data/final_excel_files/cb513_final.xlsx")
casp12_df = load_dataset("/Data/deeksha/pssp/ProtTrans/data/final_excel_files/casp12_final.xlsx")
ts115_df = load_dataset("/Data/deeksha/pssp/ProtTrans/data/final_excel_files/ts115_final.xlsx")

cb513_df['input_x'] = [re.sub(r"[UZOB]", "X", seq) for seq in cb513_df['input_x']]
casp12_df['input_x'] = [re.sub(r"[UZOB]", "X", seq) for seq in casp12_df['input_x']]
ts115_df['input_x'] = [re.sub(r"[UZOB]", "X", seq) for seq in ts115_df['input_x']]

cb513_X = cb513_df['input_x'].values
cb513_y = cb513_df[' dssp3'].values
casp12_X = casp12_df['input_x'].values
casp12_y = casp12_df[' dssp3'].values
ts115_X = ts115_df['input_x'].values
ts115_y = ts115_df[' dssp3'].values

cb513_X = [encode_sequence(sample, char2int) for sample in cb513_X]
cb513_Y = [encode_sequence(sample, label2int) for sample in cb513_y]
casp12_X = [encode_sequence(sample, char2int) for sample in casp12_X]
casp12_Y = [encode_sequence(sample, label2int) for sample in casp12_y]
ts115_X = [encode_sequence(sample, char2int) for sample in ts115_X]
ts115_Y = [encode_sequence(sample, label2int) for sample in ts115_y]

cb513_ds = list(zip(cb513_X, cb513_Y))
casp12_ds = list(zip(casp12_X, casp12_Y))
ts115_ds = list(zip(ts115_X, ts115_Y))

cb513_dl = DataLoader(cb513_ds, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
casp12_dl = DataLoader(casp12_ds, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
ts115_dl = DataLoader(ts115_ds, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)

def train(model, train_dl, epochs):
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_dl):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(label2int)), targets.view(-1))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_dl)}, Loss: {loss.item()}')



def generate_samples(num_samples, seq_length):
    data = []
    chars = list(char2int.keys())
    labels = list(label2int.keys())
    
    for _ in range(num_samples):
        primary_seq = ' '.join(random.choice(chars) for _ in range(seq_length))
        predicted_seq = ' '.join(random.choice(labels) for _ in range(seq_length))
        actual_seq = ' '.join(random.choice(labels) for _ in range(seq_length))
        data.append((primary_seq, predicted_seq, actual_seq))
    
    with open('data.txt', 'w') as f:
        for sample in data:
            f.write(f'{sample[0]}\n{sample[1]}\n{sample[2]}\n\n')
    
    return data

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[int2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [int2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

def evaluate(model, test_dl):

    model.eval()
    all_true_preds, all_true_labels, total_loss= [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)
            true_labels, true_preds = postprocess(preds, targets)
            loss = criterion(outputs.view(-1, len(int2label)), targets.view(-1))
            total_loss.append(loss.item())
            all_true_preds.extend(true_preds)
            all_true_labels.extend(true_labels)
            metric.add_batch(predictions=true_preds, references=true_labels)
    results = metric.compute()
    print(results, 'Loss:', sum(total_loss)/len(total_loss))
    # print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    # print(f'F1-score: {f1_score(y_true, y_pred, average="macro")}')

    # print(generate_samples(10, 700))

    
train(model, train_dl, 10)
evaluate(model, test_dl)
evaluate(model, cb513_dl)
evaluate(model, casp12_dl)
evaluate(model, ts115_dl)