from dataset import Casp, Cb513, Ts115, RelationalCaspDataset, RelationalCb513Dataset, RelationalTs115Dataset
import torch
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader 
import argparse
import warnings
import transformers
import evaluate
import pandas as pd
import logging
from model import (LMFreezeConcatModel, LmConcatGCNModel, GATLmConcat, GATLmFreezeConcat, BaselineModel, LmConcatRGCNModel, LmConcatGCNBLSTMModel, LmConcatRGCNModelAttn, LmConcatRGATModel
                   ,LMBaseModel, RGCNConcatModel, HalfGCNModel)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

int2primary = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
# second2int = {'C': 0, 'E': 1, 'H': 2}
# int2second = {0: 'C', 1: 'E', 2: 'H'}
int2second={0: 'B', 1: 'C', 2: 'E', 3: 'G', 4: 'H', 5: 'I', 6: 'S', 7: 'T'}
second2int={'B': 0, 'C': 1, 'E': 2, 'G': 3, 'H': 4, 'I': 5, 'S': 6, 'T': 7}

model_name = "yarongef/DistilProtBert"
tokenizer_name = 'yarongef/DistilProtBert'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def get_datasets(args):
    if args.is_relational==1:
        casp_ds, cb513_ds, ts115_ds = RelationalCaspDataset(), RelationalCb513Dataset(), RelationalTs115Dataset()
        return casp_ds, cb513_ds, ts115_ds
    else:
        cb513, casp, ts115 = Casp(), Cb513(), Ts115()
        return cb513, casp, ts115
# cb513, casp, ts115 = RelationalCb513Dataset(), RelationalCaspDataset(), RelationalTs115Dataset()

def load_data(args, ds):
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    return dl

metric = evaluate.load("seqeval")
warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()


# model_mn_pth1 = "lm_rgcn_concat"
# model_mn_pth2 = "lm_rgcn_concat_l4"
# model_mn_pth3 = "lm_rgcn_concat_l4_ln"
# model_mn_pth4 = "lm_rgcn_concat_l4_ln2"
# model_mn_pth5 = "lm_rgcn_attn"

# logging.basicConfig(level=logging.INFO, filename=f'/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/metrics/lm_rgcn_concat.log', filemode='w')

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # lm_base_model, rgcn_model = LMBaseModel(model_name).to(device), RGCNConcatModel(num_rgcn_layers=args.num_rgcn_layers).to(device)
    lm_base_model, gcn_model = LMBaseModel(model_name).to(device), HalfGCNModel(num_gcn_layers=args.num_gcn_layers).to(device)
    
    if load:
        model5.load_state_dict(torch.load(load), strict=False)
    return lm_base_model, gcn_model

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

# def test(args, _, model, eval_loader, device):
#     #  model.eval()
#      lm_model, rgcn_model = model
#      lm_model.eval()
#      rgcn_model.eval()
#      all_true_preds, all_true_labels = [], []
#      total_loss = 0
#      for batch in tqdm(eval_loader): 
#          batch.seq_encodings = {k: v.view(args.batch_size, -1).to(device) for k, v in batch.seq_encodings.items()}
#          batch = batch.to(device)
#          lm_output = lm_model(batch)
#          output = rgcn_model(batch, lm_output['logits'])
#          loss, logits = output['loss'], output['logits']
#          total_loss += loss.item() 
#          preds = logits.argmax(dim=-1)
#          true_labels, true_preds = postprocess(preds, batch.seq_encodings['labels'])
#          all_true_preds.extend(true_preds)
#          all_true_labels.extend(true_labels)
#          metric.add_batch(predictions=true_preds, references=true_labels)
     
#      results = metric.compute()
#      all_true_preds = [item for sublist in all_true_preds for item in sublist]
#      all_true_labels = [item for sublist in all_true_labels for item in sublist]

#      # Improve the appearance of the confusion matrix
#      cf = confusion_matrix(all_true_labels, all_true_preds, labels=list(int2second.values()))
#      disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=list(int2second.values()))

#      fig, ax = plt.subplots(figsize=(10, 10))
#      disp.plot(ax=ax, cmap=plt.cm.Blues)
#      plt.title('Confusion Matrix')
#      plt.savefig('/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/metrics/lm_rgcn_concat.png', bbox_inches='tight')
 
#      return results['overall_accuracy'], results['overall_f1'], results['overall_precision'], results['overall_recall'], total_loss / len(eval_loader)

def test(args, _, model, eval_loader, device):
    lm_model, rgcn_model = model
    lm_model.eval()
    rgcn_model.eval()
    all_true_preds, all_true_labels = [], []
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
        all_true_preds.extend(true_preds)
        all_true_labels.extend(true_labels)
        metric.add_batch(predictions=true_preds, references=true_labels)
     
    results = metric.compute()
    all_true_preds = [item for sublist in all_true_preds for item in sublist]
    all_true_labels = [item for sublist in all_true_labels for item in sublist]

    cf = confusion_matrix(all_true_labels, all_true_preds, labels=list(int2second.values()))
    
    # Calculate class accuracies
    import numpy as np
    cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    cf = np.around(cf, decimals=2)
    
    # Improve the appearance of the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=list(int2second.values()))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    
    plt.title('Confusion Matrix with Class Accuracies')
    if len(eval_loader) == 17:
        plt.savefig('/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/metrics/casp_q8_rgcn.png', bbox_inches='tight')
    elif len(eval_loader) >= 400:
        plt.savefig('/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/metrics/cb513_q8_rgcn.png', bbox_inches='tight')
    else:
        plt.savefig('/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/metrics/ts115_q8_rgcn.png', bbox_inches='tight')

    return results['overall_accuracy'], results['overall_f1'], results['overall_precision'], results['overall_recall'], total_loss / len(eval_loader)


def test_baseline_lm(args, _, model, eval_loader, device):
    #  model.eval()
     lm_model= model
     lm_model.eval()

     total_loss = 0
     for batch in tqdm(eval_loader): 
         batch.seq_encodings = {k: v.view(args.batch_size, -1).to(device) for k, v in batch.seq_encodings.items()}
         batch = batch.to(device)
         output = lm_model(batch)
         loss, logits = output['loss'], output['logits']
         total_loss += loss.item() 
         preds = logits.argmax(dim=-1)
         true_labels, true_preds = postprocess(preds, batch.seq_encodings['labels'])
         metric.add_batch(predictions=true_preds, references=true_labels)
     results = metric.compute()
     return results['overall_accuracy'], results['overall_f1'], results['overall_precision'], results['overall_recall'], total_loss / len(eval_loader)

def sample(args, model, val_dataset, device, ds_type):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    primary_seqs, predicted_secondary, actual_secondary = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader): 
            batch.seq_encodings = {k: v.view(args.batch_size, -1).to(device) for k, v in batch.seq_encodings.items()}
            batch = batch.to(device)
            output = model(batch)
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

    with open(args.samples_file, 'w') as f:
        f.write(f'{ds_type}\n\n')
        for i in range(len(predicted_secondary)):
            f.write(f'Primary: {"".join(primary_seqs[i])}\n')
            f.write(f'Predicted: {" ".join(predicted_secondary[i])}\n')
            f.write(f'Actual: {" ".join(actual_secondary[i])}\n\n')
            print()

def evaluation_baseline_lm(args, model):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # model.eval()

    casp, cb513, ts115 = get_datasets(args)
    casp_dl = load_data(args, casp)
    cb513_dl = load_data(args, cb513)
    ts115_dl = load_data(args, ts115)
    
    casp_acc, casp_f1, casp_pre, casp_recall, casp_loss = test_baseline_lm(args, 0, model, casp_dl, device)
    cb513_acc, cb513_f1, cb513_pre, cb513_recall, cb513_loss = test_baseline_lm(args, 0, model, cb513_dl, device)
    ts115_acc, ts115_f1, ts115_pre, ts115_recall, ts115_loss = test_baseline_lm(args, 0, model, ts115_dl, device)
    
    print(f'CASP Accuracy: {casp_acc}, CASP F1: {casp_f1}, CASP Precision: {casp_pre}, CASP Recall: {casp_recall}, CASP Loss: {casp_loss}')
    print(f'CB513 Accuracy: {cb513_acc}, CB513 F1: {cb513_f1}, CB513 Precision: {cb513_pre}, CB513 Recall: {cb513_recall}, CB513 Loss: {cb513_loss}')
    print(f'TS115 Accuracy: {ts115_acc}, TS115 F1: {ts115_f1}, TS115 Precision: {ts115_pre}, TS115 Recall: {ts115_recall}, TS115 Loss: {ts115_loss}')

    logging.info(f'CASP Accuracy: {casp_acc}, CASP F1: {casp_f1}, CASP Precision: {casp_pre}, CASP Recall: {casp_recall}, CASP Loss: {casp_loss}')
    logging.info(f'CB513 Accuracy: {cb513_acc}, CB513 F1: {cb513_f1}, CB513 Precision: {cb513_pre}, CB513 Recall: {cb513_recall}, CB513 Loss: {cb513_loss}')
    logging.info(f'TS115 Accuracy: {ts115_acc}, TS115 F1: {ts115_f1}, TS115 Precision: {ts115_pre}, TS115 Recall: {ts115_recall}, TS115 Loss: {ts115_loss}')

    data = {'Dataset': ['CASP', 'CB513', 'TS115'], 'Accuracy': [casp_acc, cb513_acc, ts115_acc], 'F1': [casp_f1, cb513_f1, ts115_f1], 'Precision': [casp_pre, cb513_pre, ts115_pre], 'Recall': [casp_recall, cb513_recall, ts115_recall], 'Loss': [casp_loss, cb513_loss, ts115_loss]}
    # df = pd.DataFrame(data)

    # sample(args, model, casp_ds, device, 'casp')
    # sample(args, model, cb513_ds, device, 'cb513')
    # sample(args, model, ts115_ds, device, 'ts115')

def evaluation(args, model):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # model.eval()

    casp, cb513, ts115 = get_datasets(args)
    casp_dl = load_data(args, casp)
    cb513_dl = load_data(args, cb513)
    ts115_dl = load_data(args, ts115)
    
    casp_acc, casp_f1, casp_pre, casp_recall, casp_loss = test(args, 0, model, casp_dl, device)
    cb513_acc, cb513_f1, cb513_pre, cb513_recall, cb513_loss = test(args, 0, model, cb513_dl, device)
    ts115_acc, ts115_f1, ts115_pre, ts115_recall, ts115_loss = test(args, 0, model, ts115_dl, device)
    
    print(f'CASP Accuracy: {casp_acc}, CASP F1: {casp_f1}, CASP Precision: {casp_pre}, CASP Recall: {casp_recall}, CASP Loss: {casp_loss}')
    print(f'CB513 Accuracy: {cb513_acc}, CB513 F1: {cb513_f1}, CB513 Precision: {cb513_pre}, CB513 Recall: {cb513_recall}, CB513 Loss: {cb513_loss}')
    print(f'TS115 Accuracy: {ts115_acc}, TS115 F1: {ts115_f1}, TS115 Precision: {ts115_pre}, TS115 Recall: {ts115_recall}, TS115 Loss: {ts115_loss}')

    logging.info(f'CASP Accuracy: {casp_acc}, CASP F1: {casp_f1}, CASP Precision: {casp_pre}, CASP Recall: {casp_recall}, CASP Loss: {casp_loss}')
    logging.info(f'CB513 Accuracy: {cb513_acc}, CB513 F1: {cb513_f1}, CB513 Precision: {cb513_pre}, CB513 Recall: {cb513_recall}, CB513 Loss: {cb513_loss}')
    logging.info(f'TS115 Accuracy: {ts115_acc}, TS115 F1: {ts115_f1}, TS115 Precision: {ts115_pre}, TS115 Recall: {ts115_recall}, TS115 Loss: {ts115_loss}')

    data = {'Dataset': ['CASP', 'CB513', 'TS115'], 'Accuracy': [casp_acc, cb513_acc, ts115_acc], 'F1': [casp_f1, cb513_f1, ts115_f1], 'Precision': [casp_pre, cb513_pre, ts115_pre], 'Recall': [casp_recall, cb513_recall, ts115_recall], 'Loss': [casp_loss, cb513_loss, ts115_loss]}
    # df = pd.DataFrame(data)

    # sample(args, model, casp_ds, device, 'casp')
    # sample(args, model, cb513_ds, device, 'cb513')
    # sample(args, model, ts115_ds, device, 'ts115')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/metrics/lm_fr_concat.pt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--samples_file', type=str, default='/Data/deeksha/pssp/ProtTrans/scripts/train/MultiModal/example_preds/lm_rgcn_concat.txt')
    parser.add_argument('--num_gcn_layers', type=int, default=2)
    parser.add_argument('--num_rgcn_layers', type=int, default=2)
    parser.add_argument('--num_gat_layers', type=int, default=2)
    parser.add_argument('--is_relational', type=int, default=1)
    parser.add_argument('--is_2lr', type=int, default=0)
    args = parser.parse_args()
    # model = load_model(args)
    '''Baseline Model'''
    model = BaselineModel(model_name).to(device)
    evaluation_baseline_lm(args, model)
    '''SSRNet-RGCN-Model'''
    # lm_base_model, rgcn_model = LMBaseModel(model_name).to(device), RGCNConcatModel(num_rgcn_layers=args.num_rgcn_layers).to(device)
    # lm_base_model.load_state_dict(torch.load('/Data/deeksha/pssp/ProtTrans/scripts/train/q8_models/lm_rgcn_q8_with2lr_lm.pt'), strict=False)
    # rgcn_model.load_state_dict(torch.load('/Data/deeksha/pssp/ProtTrans/scripts/train/q8_models/lm_rgcn_q8_with2lr_rgcn.pt'), strict=False)
    
    '''SSRNet-GCN-Model'''
    # lm_base_model, gcn_model = LMBaseModel(model_name).to(device), HalfGCNModel(num_gcn_layers=args.num_gcn_layers).to(device)
    # lm_base_model.load_state_dict(torch.load('/Data/deeksha/pssp/ProtTrans/scripts/train/q8_models/lm_gcn_q8_with2lr_lm.pt'), strict=False)
    # gcn_model.load_state_dict(torch.load('/Data/deeksha/pssp/ProtTrans/scripts/train/q8_models/lm_gcn_q8_with2lr_rgcn.pt'), strict=False)
    # model = (lm_base_model, gcn_model)
    # evaluation(args, model)
