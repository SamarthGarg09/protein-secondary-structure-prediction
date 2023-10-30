import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from transformers import AutoTokenizer
from torch_geometric.nn import GCNConv, RGCNConv, RGATConv
from torch_geometric.nn.models import GAT
import torch
import torch.nn.functional as F
from torch import nn

# second2int = {'C': 0, 'E': 1, 'H': 2}
# int2second = {0: 'C', 1: 'E', 2: 'H'}
int2second={0: 'B', 1: 'C', 2: 'E', 3: 'G', 4: 'H', 5: 'I', 6: 'S', 7: 'T'}
second2int={'B': 0, 'C': 1, 'E': 2, 'G': 3, 'H': 4, 'I': 5, 'S': 6, 'T': 7}

model_name = "yarongef/DistilProtBert"
tokenizer_name = 'yarongef/DistilProtBert'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# lm_concat model
class LmConcatGCNModel(torch.nn.Module):
    def __init__(self, model_name, num_gcn_layers, hidden_dim=128, num_classes=8) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        # create the lm head for token classification
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.gcnModel = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        # self.cls_head = GCNConv(hidden_dim, h)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        # get the last hidden state
        last_hidden_state = output.last_hidden_state
        output = self.dropout(last_hidden_state)
        lm_output = self.lm_head(output)
        output = lm_output.clone()
        output = output.view(-1, 128)
        for conv in self.gcnModel:
            output = conv(output, batch.edge_index)
            output = F.relu(output)
            output = self.dropout(output)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output
    
# lm_fr_concat model
class LMFreezeConcatModel(torch.nn.Module):
    def __init__(self, model_name, num_gcn_layers, hidden_dim=128, num_classes=8) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        # create the lm head for token classification
        self.lm_head = nn.Linear(1024, hidden_dim)
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.gcnModel = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        # self.cls_head = GCNConv(hidden_dim, h)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        with torch.no_grad():
            output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        # get the last hidden state
        last_hidden_state = output.last_hidden_state
        output = self.dropout(last_hidden_state)
        lm_output = self.lm_head(output)
        output = lm_output.clone()
        for conv in self.gcnModel:
            output = conv(output, batch.edge_index)
            output = F.relu(output)
            output = self.dropout(output)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output


class GATLmConcat(torch.nn.Module):
    def __init__(self, model_name, num_gat_layers, hidden_dim=128, num_classes=8) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.gat_model = nn.ModuleList([
            GAT(hidden_dim, hidden_dim, num_layers=num_gat_layers, dropout=0.1)
            for _ in range(num_gat_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        # get the last hidden state
        last_hidden_state = output.last_hidden_state
        batch_size= last_hidden_state.shape[0]
        output = self.dropout(last_hidden_state)
        lm_output = self.lm_head(output)
        output = lm_output.clone()
        output = output.view(-1, 128)
        for layer in self.gat_model:
            output = layer(output, batch.edge_index)
            output = F.relu(output)
            output = self.dropout(output)
        # output = self.gat_model(output, batch.edge_index)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

# GATLmFreezeConcatGCN
class GATLmFreezeConcat(torch.nn.Module):
    def __init__(self, model_name, num_gat_layers, hidden_dim=128, num_classes=8) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        # freeze all the parameters of the language model exept the last layer
        for name, param in self.language_model.named_parameters():
            if name.startswith('bert.encoder.layer.14'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        # create the lm head for token classification
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.gat_model = GAT(hidden_dim, hidden_dim, num_layers=num_gat_layers, dropout=0.1)

        # self.cls_head = GCNConv(hidden_dim, h)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        with torch.no_grad():
            output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        # get the last hidden state
        last_hidden_state = output.last_hidden_state
        batch_size= last_hidden_state.shape[0]
        output = self.dropout(last_hidden_state)
        lm_output = self.lm_head(output)
        output = lm_output.clone()
        output = output.view(-1, 128)
        output = self.gat_model(output, batch.edge_index)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

class BaselineModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=8, 
            id2label=int2second,
            label2id=second2int
            )
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, batch):       
        output = self.model(
            input_ids=batch.seq_encodings['input_ids'], 
            attention_mask=batch.seq_encodings['attention_mask'], 
            token_type_ids=batch.seq_encodings['token_type_ids'],
            labels=batch.seq_encodings['labels']
            )
        return {
            'loss': output.loss,
            'logits': output.logits
        }

class RGCNModel(torch.nn.Module):
    def __init__(self, fan_in, fan_out, num_relations, num_rgcn_layers, dropout_ratio=0.4):
        super(RGCNModel, self).__init__()
        
        self.conv_layers = torch.nn.ModuleList([
            RGCNConv(fan_in, fan_out, num_relations)
            for _ in range(num_rgcn_layers)
        ])
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x, edge_index, edge_type):
        
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x


class LmConcatRGCNModel(torch.nn.Module):
    def __init__(self, model_name, num_relations=3, num_rgcn_layers=2, hidden_dim=128, num_classes=8, dropout_ratio=0.4) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.lm_layer_norm = nn.LayerNorm(hidden_dim)  # Layer Normalization layer
        self.rgcnModel = RGCNModel(fan_in=hidden_dim, fan_out=hidden_dim, num_relations=num_relations, num_rgcn_layers=num_rgcn_layers)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        
        output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        output = self.dropout(F.relu(output.last_hidden_state))
        lm_output = self.dropout(F.relu(self.lm_head(output)))
        lm_output = self.lm_layer_norm(lm_output)  # Apply Layer Normalization

        output = lm_output.clone()
        output = output.view(-1, 128)
        output = self.rgcnModel(output, batch.edge_index.long(), batch.edge_type)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

class LmConcatGCNBLSTMModel(torch.nn.Module):
    def __init__(self, model_name, num_gcn_layers, hidden_dim=128, num_classes=8) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        # create the lm head for token classification
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.gcnModel = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        # self.cls_head = GCNConv(hidden_dim, h)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.blstm = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        # get the last hidden state
        last_hidden_state = output.last_hidden_state
        output = self.dropout(last_hidden_state)
        lm_output = self.lm_head(output)
        output = lm_output.clone()
        output = output.view(-1, 128)
        for conv in self.gcnModel:
            output = conv(output, batch.edge_index)
            output = F.relu(output)
            output = self.dropout(output)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output, _ = self.blstm(output)
        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

class LmConcatRGCNModelAttn(torch.nn.Module):
    def __init__(self, model_name, num_relations=3, num_rgcn_layers=2, hidden_dim=128, num_classes=8, dropout_ratio=0.4) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        self.language_model.resize_token_embeddings(len(tokenizer))
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.lm_layer_norm = nn.LayerNorm(hidden_dim)  
        self.rgcnModel = RGCNModel(fan_in=hidden_dim, fan_out=hidden_dim, num_relations=num_relations, num_rgcn_layers=num_rgcn_layers)

        self.attn = nn.MultiheadAttention(batch_first=True, embed_dim=hidden_dim, num_heads=4)
        self.attn_fc = nn.Linear(hidden_dim, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        
        output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        output = self.dropout(F.relu(output.last_hidden_state))
        lm_output = self.dropout(F.relu(self.lm_head(output)))
        lm_output = self.lm_layer_norm(lm_output)  

        output = lm_output.clone()
        output = output.view(-1, 128)
        output = self.rgcnModel(output, batch.edge_index.long(), batch.edge_type)
        output = output.view(batch_size, -1, 128)

        output = self.attn_fc(output)
        output, _ = self.attn(output, lm_output, lm_output)
        output = self.attn_fc(output)
        output = output + lm_output

        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output


class RGAT(nn.Module):
    def __init__(self, fan_in, fan_out, num_relations, num_rgat_layers, dropout_ratio=0.4):
        super().__init__()

        self.rgat_conv=nn.ModuleList([RGATConv(fan_in,fan_out,num_relations,concat=False) 
                                     for _ in range(num_rgat_layers)])
        self.dropout=nn.Dropout(dropout_ratio)

    def forward(self,x, edge_index, edge_type):
        for conv in self.rgat_conv:
            x=conv(x,edge_index,edge_type)
            x=F.relu(x)
            x=self.dropout(x)
        return x

class LmConcatRGATModel(torch.nn.Module):
    def __init__(self, model_name, num_relations=3, num_rgat_layers=2, hidden_dim=128, num_classes=8, dropout_ratio=0.4) -> None:
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)
        

        self.language_model.resize_token_embeddings(len(tokenizer))
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.lm_layer_norm = nn.LayerNorm(hidden_dim)  # Layer Normalization layer
        self.rgatModel = RGAT(fan_in=hidden_dim, fan_out=hidden_dim, num_relations=num_relations, num_rgat_layers=num_rgat_layers)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        
        output = self.language_model(input_ids=batch.seq_encodings['input_ids'], attention_mask=batch.seq_encodings['attention_mask'], token_type_ids=batch.seq_encodings['token_type_ids'])
        output = self.dropout(F.relu(output.last_hidden_state))
        lm_output = self.dropout(F.relu(self.lm_head(output)))
        lm_output = self.lm_layer_norm(lm_output)  # Apply Layer Normalization

        output = lm_output.clone()
        output = output.view(-1, 128)
        output = self.rgatModel(output, batch.edge_index.long(), batch.edge_type)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

class BaselineModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=8, 
            id2label=int2second,
            label2id=second2int
            )
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, batch):       
        output = self.model(
            input_ids=batch.seq_encodings['input_ids'], 
            attention_mask=batch.seq_encodings['attention_mask'], 
            token_type_ids=batch.seq_encodings['token_type_ids'],
            labels=batch.seq_encodings['labels']
            )
        return {
            'loss': output.loss,
            'logits': output.logits
        }
class LMBaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.language_model = AutoModel.from_pretrained(model_name, 
                                                        num_labels=8,
                                                        id2label=int2second,
                                                        label2id=second2int)

        self.language_model.resize_token_embeddings(len(tokenizer))

    def forward(self, batch):       
        output = self.language_model(
            input_ids=batch.seq_encodings['input_ids'], 
            attention_mask=batch.seq_encodings['attention_mask'], 
            token_type_ids=batch.seq_encodings['token_type_ids'],
            )
        return {
            'logits': output.last_hidden_state
        }


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(hidden_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        # Query, Key, Value linear layers
        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.out = nn.Linear(hidden_dim, hidden_dim)

    def transpose_for_scores(self, x):
        return x.view(-1, self.num_heads, self.attention_head_size).transpose(0, 1)

    def forward(self, lm_output, graph_output):
        # Compute Q, K, V values
        query_layer = self.transpose_for_scores(self.query(lm_output))
        key_layer = self.transpose_for_scores(self.key(graph_output))
        value_layer = self.transpose_for_scores(self.value(graph_output))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 1).contiguous().view(-1, self.all_head_size)

        # Linear layer
        attention_output = self.out(context_layer)

        return attention_output

class CrossModelAttention(nn.Module):
    def __init__(self, num_relations=3, num_rgcn_layers=2, hidden_dim=128, num_classes=8, dropout_ratio=0.4) -> None:
        super().__init__()
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.lm_layer_norm = nn.LayerNorm(hidden_dim)  # Layer Normalization layer
        self.rgcnModel = RGCNModel(fan_in=hidden_dim, fan_out=hidden_dim, num_relations=num_relations, num_rgcn_layers=num_rgcn_layers)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads=num_classes)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Batch Normalization layer
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch, output):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        
        lm_output = self.dropout(F.relu(self.lm_head(output)))
        lm_output = self.lm_layer_norm(lm_output)  # Apply Layer Normalization

        graph_output = lm_output.clone()
        graph_output = graph_output.view(-1, 128)
        graph_output = self.rgcnModel(graph_output, batch.edge_index.long(), batch.edge_type)
        graph_output = graph_output.view(batch_size, -1, 128)

        fused_output = self.multihead_attention(lm_output, graph_output)
        # print('lm_output_before: ', lm_output.shape)
        # print("before", fused_output.shape)#(1024, 128)

        fused_output = self.batch_norm((fused_output + lm_output).permute(0, 2, 1)).permute(0, 2, 1)  # Residual connection and Batch Normalization
        # print("before", fused_output.shape)
        fused_output = self.dropout(fused_output)  # Dropout for regularization

        output = self.cls_head(fused_output)
        
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

class SeriesModelAttention(nn.Module):
    def __init__(self, num_relations=3, num_rgcn_layers=2, hidden_dim=128, num_classes=8, dropout_ratio=0.4):
        super().__init__()
        
        # Language Model Head
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        
        # RGCN Model
        self.rgcnModel = RGCNModel(fan_in=hidden_dim, fan_out=hidden_dim, num_relations=num_relations, num_rgcn_layers=num_rgcn_layers)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Classifier Head
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        
        # Loss Function
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.num_labels = num_classes
        
        # Initialize Weights
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch, output):
        hidden_dim = 128
        batch_size = batch.seq_encodings['input_ids'].size(0)
        
        # Process through LM head
        lm_output = self.dropout(F.relu(self.lm_head(output)))
        
        # Process through RGCN
        graph_output = lm_output.view(-1, hidden_dim)
        graph_output = self.rgcnModel(graph_output, batch.edge_index.long(), batch.edge_type)
        graph_output = graph_output.view(batch_size, -1, hidden_dim)
        
        # Combine outputs
        fused_output = graph_output + lm_output
        fused_output = self.batch_norm(fused_output.permute(0, 2, 1)).permute(0, 2, 1)
        fused_output = self.dropout(fused_output)
        
        # Classifier
        logits = self.cls_head(fused_output)
        
        # Calculate Loss if labels are provided
        if batch.seq_encodings['labels'] is not None:
            active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_function.ignore_index).type_as(batch.seq_encodings['labels'])
            )
            loss = self.loss_function(active_logits, active_labels)
            return {'loss': loss, 'logits': logits}
        
        return logits



class RGCNConcatModel(nn.Module):
    def __init__(self, num_relations=3, num_rgcn_layers=2, hidden_dim=128, num_classes=8, dropout_ratio=0.4) -> None:
        super().__init__()
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.lm_layer_norm = nn.LayerNorm(hidden_dim)  # Layer Normalization layer
        self.rgcnModel = RGCNModel(fan_in=hidden_dim, fan_out=hidden_dim, num_relations=num_relations, num_rgcn_layers=num_rgcn_layers)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch, output):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        
        lm_output = self.dropout(F.relu(self.lm_head(output)))
        lm_output = self.lm_layer_norm(lm_output)  # Apply Layer Normalization

        output = lm_output.clone()
        output = output.view(-1, 128)
        output = self.rgcnModel(output, batch.edge_index.long(), batch.edge_type)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output

# lm_concat model
class HalfGCNModel(torch.nn.Module):
    def __init__(self, num_gcn_layers, hidden_dim=128, num_classes=8) -> None:
        super().__init__()
        # create the lm head for token classification
        self.lm_head = nn.Linear(1024, hidden_dim)
        self.gcnModel = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        # self.cls_head = GCNConv(hidden_dim, h)
        self.cls_head = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_classes
        self.loss_fucntion = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, batch, output):
        batch_size = batch.seq_encodings['input_ids'].size(0)
        lm_output = self.lm_head(output)
        output = lm_output.clone()
        output = output.view(-1, 128)
        for conv in self.gcnModel:
            output = conv(output, batch.edge_index.long())
            output = F.relu(output)
            output = self.dropout(output)
        output = output.view(batch_size, -1, 128)
        output = torch.cat((lm_output, output), dim=-1)
        output = self.cls_head(output)
        if batch.seq_encodings['labels'] is not None:
            if batch.seq_encodings['attention_mask'] is not None:
                active_loss = batch.seq_encodings['attention_mask'].view(-1) == 1
                active_logits = output.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, batch.seq_encodings['labels'].view(-1), torch.tensor(self.loss_fucntion.ignore_index).type_as(batch.seq_encodings['labels'])
                )
                loss = self.loss_fucntion(active_logits, active_labels)
            return {
                'loss': loss,
                'logits': output
            }
        return output
