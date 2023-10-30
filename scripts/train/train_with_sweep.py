import torch
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging
from transformers import AutoTokenizer
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader 
from early_stopping import EarlyStopping
import argparse
import warnings
import transformers
import evaluate
from evaluation import evaluation

metric = evaluate.load("seqeval")
warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, filename='/Data/deeksha/disha/ProtTrans/scripts/train/MultiModal/metrics/lm_fr_concat.log', filemode='w')

# second2int = {'C': 0, 'E': 1, 'H': 2}
# int2second = {0: 'C', 1: 'E', 2: 'H'}
int2second={0: 'B', 1: 'C', 2: 'E', 3: 'G', 4: 'H', 5: 'I', 6: 'S', 7: 'T'}
second2int={'B': 0, 'C': 1, 'E': 2, 'G': 3, 'H': 4, 'I': 5, 'S': 6, 'T': 7}

model_name = "yarongef/DistilProtBert"


tokenizer_name = 'yarongef/DistilProtBert'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
from dataset import ProteinDataset, ProteinDatasetForRGCN
from model import (LMFreezeConcatModel, LmConcatGCNModel, GATLmConcat, GATLmFreezeConcat, BaselineModel, LmConcatRGCNModel, LmConcatGCNBLSTMModel, LmConcatRGCNModelAttn, LmConcatRGATModel
                   ,LMBaseModel, RGCNConcatModel, HalfGCNModel)

def load_data(args):

    # dataset = ProteinDataset()
    dataset = ProteinDatasetForRGCN()
    from sklearn.model_selection import train_test_split
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.05, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, val_dataset

device = "cuda"if torch.cuda.is_available() else "cpu"

def load_model(args, load=None):
    # model = MultiModal(model_name).to(device)
    # baseline_model = BaselineModel(model_name).to(device)
    # model1 = LMFreezeConcatModel(model_name, num_gcn_layers=args.num_gcn_layers).to(device)
    # model2 = LmConcatGCNModel(model_name, num_gcn_layers=args.num_gcn_layers).to(device)
    # model3 = GATLmConcat(model_name, num_gat_layers=args.num_gat_layers).to(device)
    # model4 = GATLmFreezeConcat(model_name, num_gcn_layers=args.num_gat_layers).to(device)
    model5 = LmConcatRGCNModel(model_name, num_rgcn_layers=args.num_rgcn_layers).to(device)
    # model6 = LmConcatGCNBLSTMModel(model_name, num_gcn_layers=args.num_gcn_layers).to(device)
    # model7 = LmConcatRGCNModelAttn(model_name, num_rgcn_layers=args.num_rgcn_layers).to(device)
    # model7 = LmConcatRGATModel(model_name, num_rgat_layers=args.num_rgat_layers).to(device)
    lm_base_model, rgcn_model = LMBaseModel(model_name).to(device), RGCNConcatModel(num_rgcn_layers=args.num_rgcn_layers).to(device)
    # lm_base_model, gcn_model = LMBaseModel(model_name).to(device), HalfGCNModel(num_gcn_layers=args.num_gcn_layers).to(device)
    if load:
        model5.load_state_dict(torch.load(load), strict=False)
    return (lm_base_model, rgcn_model )

def load_configs(args, model, train_loader):
    lm_model, rgcn_model = model
    optimizer_lm = torch.optim.Adam(lm_model.parameters(), lr=args.lm_lr, weight_decay=0.0)
    optimizer_rgcn = torch.optim.Adam(rgcn_model.parameters(), lr=args.rgcn_lr, weight_decay=0.0)
    from transformers import get_linear_schedule_with_warmup
    if args.grad_acc_steps > 1:
        total_steps = len(train_loader) * args.epochs // args.grad_acc_steps
    else:
        total_steps = len(train_loader) * args.epochs
    scheduler_lm = get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps=args.num_warmup_steps_lm, num_training_steps=total_steps)
    scheduler_rgcn = get_linear_schedule_with_warmup(optimizer_rgcn, num_warmup_steps=args.num_warmup_steps_rgcn, num_training_steps=total_steps)
    return (optimizer_lm, optimizer_rgcn), (scheduler_lm, scheduler_rgcn)

#################################################################
################## TRAIN ########################################
#################################################################

def train_epoch(args, epoch, model, train_loader, optimizer, scheduler, device):
    lm_model, rgcn_model = model
    lm_model.train()
    rgcn_model.train()

    total_loss = 0
    optimizer_lm, optimizer_rgcn = optimizer
    scheduler_lm, scheduler_rgcn = scheduler
    for step, batch in enumerate(tqdm(train_loader)): 
         if step == 0:
            optimizer_lm.zero_grad()
            optimizer_rgcn.zero_grad()
         batch.seq_encodings = {k: v.view(args.batch_size, -1).to(device) for k, v in batch.seq_encodings.items()}
        #  put edge features and other batch features to cuda
         batch = batch.to(device)

         lm_output = lm_model(batch)
         output = rgcn_model(batch, lm_output['logits'])
         loss, logits = output['loss'], output['logits']
         if args.grad_acc_steps > 1:
            loss = loss / args.grad_acc_steps
         total_loss += loss.item()
         loss.backward()
         wandb.log({"loss": loss.item()})
        #  apply gradient accumulation
         if args.grad_acc_steps > 1:
            if step % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        #  if step % args.grad_acc_steps == 0:
            
         optimizer_lm.step()
         optimizer_rgcn.step()
         optimizer_lm.zero_grad()
         optimizer_rgcn.zero_grad()

         scheduler_lm.step()
         scheduler_rgcn.step()

         preds = logits.argmax(dim=-1)
         true_labels, true_preds = postprocess(preds, batch.seq_encodings['labels'])
         metric.add_batch(predictions=true_preds, references=true_labels)
    results = metric.compute()
    wandb.log(
        {
            "train_loss": total_loss / len(train_loader),
            "train_acc": results['overall_accuracy'],
            "train_f1": results['overall_f1'],
            "train_pre": results['overall_precision'],
            "train_recall": results['overall_recall']
        }
    )
    return results['overall_accuracy'], results['overall_f1'], results['overall_precision'], results['overall_recall'], total_loss / len(train_loader)


def test(args, epoch, model, eval_loader, device):
     lm_model, rgcn_model = model
     lm_model.eval()
     rgcn_model.eval()

     total_loss = 0
     for batch in tqdm(eval_loader): 
         batch.seq_encodings = {k: v.view(args.batch_size, -1).to(device) for k, v in batch.seq_encodings.items()}
         batch = batch.to(device)
         lm_output = lm_model(batch)
         output = rgcn_model(batch, lm_output['logits'])
         loss, logits = output['loss'], output['logits']
         total_loss += loss.item() 
         preds = logits.argmax(dim=-1)
         true_labels, true_preds = postprocess(preds, batch.seq_encodings['labels'])
         metric.add_batch(predictions=true_preds, references=true_labels)
         wandb.log({"loss": loss.item()})
     results = metric.compute()

     wandb.log(
        {
            "test_loss": total_loss / len(eval_loader),
            "test_acc": results['overall_accuracy'],
            "test_f1": results['overall_f1'],
            "test_pre": results['overall_precision'],
            "test_recall": results['overall_recall']
        }
    )
     return results['overall_accuracy'], results['overall_f1'], results['overall_precision'], results['overall_recall'], total_loss / len(eval_loader)


# loss and train lists
def train(args, model, train_loader, val_loader, val_dataset, optimizer, scheduler, patience, delta, device):
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta, path=args.model_path, is_2lr=args.is_2lr)
    lm_model, rgcn_model = model
    wandb.watch(lm_model, log='all', log_freq=2000)
    wandb.watch(rgcn_model, log='all', log_freq=2000)
    for epoch in range(args.epochs):
        train_acc, train_f1, train_pre, train_recall, train_loss =  train_epoch(args, epoch, model, train_loader, optimizer, scheduler, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} :")
        print(f"Train loss : {train_loss:.4f}, Train acc : {train_acc:.4f}, Train f1 : {train_f1:.4f}, Train pre : {train_pre:.4f}, Train recall : {train_recall:.4f}")
        test_acc, test_f1, test_pre, test_recall, test_loss = test(args, epoch, model, val_loader, device)
        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        
        print(f"Test loss : {test_loss:.4f}, Test acc : {test_acc:.4f}, Test f1 : {test_f1:.4f}, Test pre : {test_pre:.4f}, Test recall : {test_recall:.4f}")
        logging.info(f"Epoch {epoch+1}/{args.epochs} :")
        logging.info(f"Test loss : {test_loss:.4f}, Test acc : {test_acc:.4f}, Test f1 : {test_f1:.4f}, Test pre : {test_pre:.4f}, Test recall : {test_recall:.4f}")
        
        if epoch % args.sample_every == 0:
            sample(args, model, val_dataset, device)
        
        early_stopping(test_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model_path = os.path.join("/Data/deeksha/disha/ProtTrans/scripts/train/q8_models", args.model_path)

    lm_model.load_state_dict(torch.load(model_path.split('.')[1] + '_lm.pt'))
    rgcn_model.load_state_dict(torch.load(model_path.split('.')[1] + '_rgcn.pt'))

    return train_acc, train_f1, train_pre, train_recall, test_acc, test_loss, test_pre, test_recall
    # return [], [], [], [], test_acc, test_loss, test_pre, test_recall

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[int2second[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [int2second[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

def sample(args, model, val_dataset, device):
    lm_model, rgcn_model = model
    lm_model.train()
    rgcn_model.train()

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    primary_seqs, predicted_secondary, actual_secondary = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader): 
            batch.seq_encodings = {k: v.view(args.batch_size, -1).to(device) for k, v in batch.seq_encodings.items()}
            batch = batch.to(device)
            lm_output = lm_model(batch)
            output = rgcn_model(batch, lm_output['logits'])
            _, logits = output['loss'], output['logits']
            preds = logits.argmax(dim=-1)
            true_labels, true_preds = postprocess(preds, batch.seq_encodings['labels'])
            primary_seq = tokenizer.batch_decode(batch.seq_encodings['input_ids'], skip_special_tokens=True)
            primary_seqs.extend(primary_seq)
            predicted_secondary.extend(true_preds)
            actual_secondary.extend(true_labels)
            if len(predicted_secondary) == args.num_samples:
                break
        predicted_secondary = [sample for sample in predicted_secondary]
        actual_secondary = [sample for sample in actual_secondary]
        primary_seqs = [sample for sample in primary_seqs]
 
    sample_file = os.path.join('/Data/deeksha/disha/ProtTrans/scripts/train/MultiModal/q8_samples', args.samples_file)
    with open(sample_file, 'w') as f:
        for i in range(len(predicted_secondary)):
            f.write(f'Primary: {"".join(primary_seqs[i])}\n')
            f.write(f'Predicted: {" ".join(predicted_secondary[i])}\n')
            f.write(f'Actual: {" ".join(actual_secondary[i])}\n\n')
            print()

def arg_parse():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--model_path', type=str, default='lm_rgcn_with2lr.pt')
    parser.add_argument('--samples_file', type=str, default='lm_rgcn_with2lr.txt')
    parser.add_argument('-e', '--epochs', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--q', type=str, default='q3')
    parser.add_argument('--grad_acc_steps', type=int, default=0)
    parser.add_argument('--num_gcn_layers', type=int, default=3)
    parser.add_argument('--num_gat_layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_rgcn_layers', type=int, default=4)
    parser.add_argument('--num_rgat_layers', type=int, default=2)
    parser.add_argument('--is_relational', type=int, default=1)
    parser.add_argument('--lm_lr', type=float, default=1e-05)
    parser.add_argument('--rgcn_lr', type=float, default=3e-04)
    parser.add_argument('--is_2lr', type=int, default=1)
    parser.add_argument('--num_warmup_steps_lm', type=int, default=150)
    parser.add_argument('--num_warmup_steps_rgcn', type=int, default=50)
    
    
    return parser.parse_args()
def main(config=None):
    with wandb.init(project="pssp_main_q8", name="rgcn_q8_sweep"):
        args = wandb.config
        train_loader, val_loader, val_dataset = load_data(args)
        print("Loading model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(args)
        print("Loading configs")
        optimizer, scheduler = load_configs(args, model, train_loader)
        print("Training")
        train(args, model, train_loader, val_loader, val_dataset, optimizer, scheduler, args.patience, args.delta, device)
        evaluation(args, model)
        print("Sampling")
        sample(args, model, val_dataset, device)
        print("Done")

if __name__ == '__main__':
    args = arg_parse()
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "test_acc",
            "goal": "maximize"
        },
        "parameters": {
            "num_samples": {
                "values": [50]
            },
            "sample_every": {
                "values": [5]
            },
            "model_path": {
                "values": ['lm_gcn_with2lr.pt']
            },
            "samples_file": {
                "values": ['lm_gcn_with2lr.txt']
            },
            "epochs": {
                "values": [3, 4, 5],
            },
            "batch_size": {
                "values": [1],
            },
            "lr": {
                "values": [3e-05]
            },
            "q": {
                "values": ['q3']
            },
            "grad_acc_steps": {
                "values": [0]
            },
            "num_gcn_layers": {
                "values": [2, 3, 4]
            },
            "num_gat_layers": {
                "values": [2]
            },
            "patience": {
                "values": [2]
            },
            "delta": {
                "values": [0.001]
            },
            "num_heads": {
                "values": [4]
            },
            "num_rgcn_layers": {
                "values": [2, 3, 4]
            },
            "num_rgat_layers": {
                "values": [2]
            },
            "is_relational": {
                "values": [1]
            },
            "lm_lr": {
                "distribution": "uniform",
                "min": 1e-5,
                "max": 1e-4
            },
            "rgcn_lr": {
                "distribution": "uniform",
                "min": 1e-4,
                "max": 1e-3
            },
            "is_2lr": {
                "values": [1]
            },
            "num_warmup_steps_lm": {
                "values": [150, 200, 300],
            },
            "num_warmup_steps_rgcn": {
                "values": [50, 100, 200],
            },
        }
    }
    print("Loading data")
    sweep_id = wandb.sweep(sweep_config, project="pssp_main_q8")
    wandb.agent(sweep_id, function=main, count=1)
    wandb.finish()

