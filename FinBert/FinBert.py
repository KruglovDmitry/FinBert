import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import statistics
from functools import partial
import torch
import pickle
import pytorch_lightning as pl
import torchmetrics
from ptls.frames.supervised import SequenceToTarget
from torch.nn import CrossEntropyLoss
from ptls.nn import TrxEncoder, Head, PBLinear, PBL2Norm, PBLayerNorm, PBDropout
from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing import SeqLenFilter, FeatureFilter, ToTorch
from ptls.frames.supervised.seq_to_target_dataset import SeqToTargetIterableDataset
from ptls.data_load import IterableChain
from ptls.frames import PtlsDataModule
from Model import MLMCPCPretrainModule, Model, eval_model
import matplotlib.pyplot as plt
from MatMulWithNoise import FuncFactory

print("Pythorch version - ", torch.__version__)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.multiprocessing.set_sharing_strategy('file_system')

# load preprocessed data:
with open('..\\finbert_data\\preproc_data', 'rb') as h:
    preproc_data = pickle.load(h)
    numeric_values = preproc_data['numeric_values']
    embeddings = preproc_data['embeddings']


trx_encoder_params = dict(
    embeddings_noise=0,
    numeric_values=numeric_values,
    embeddings=embeddings,
    emb_dropout=0.3,
    spatial_dropout=False)
trx_encoder = TrxEncoder(**trx_encoder_params)

train_pq_files = ParquetFiles('..\\finbert_data\\train.parquet\\')
valid_pq_files = ParquetFiles('..\\finbert_data\\valid.parquet\\')

train_dataset = ParquetDataset(data_files=train_pq_files.data_files, shuffle_files=True)
valid_dataset = ParquetDataset(data_files=valid_pq_files.data_files, shuffle_files=True)

finetune_dm = PtlsDataModule(
    train_data=SeqToTargetIterableDataset(train_dataset, target_col_name='flag'),
    valid_data=SeqToTargetIterableDataset(valid_dataset, target_col_name='flag'),
    train_num_workers=0, #20
    train_batch_size=1024,
    valid_batch_size=1024,)

seq_encoder = MLMCPCPretrainModule(
    trx_encoder=torch.nn.Sequential(
        trx_encoder,
        PBLinear(trx_encoder.output_size, 64),
        PBDropout(0.2)
    ),
    hidden_size=64,
    loss_temperature=20.0,
    total_steps=30000,
    replace_proba=0.1,
    neg_count=64,
    log_logits=False,
    encode_seq=True,)

model = Model(seq_encoder, 
              Head(
                input_size=64,
                hidden_layers_sizes=[32, 8],
                drop_probs=[0.1, 0],
                use_batch_norm=True,
                objective='classification',
                num_classes=2,
                ),
              torch.nn.NLLLoss())

TOTAL_EPOCHS = 2
MAX_ITERS = 3000 # Total 3000000 uniqe id
EVAL_INTERVAL = 200

train_dl = finetune_dm.train_dl(SeqToTargetIterableDataset(train_dataset, target_col_name='flag'))
val_dl = finetune_dm.val_dl(SeqToTargetIterableDataset(valid_dataset, target_col_name='flag'))
optimizer = torch.optim.AdamW(seq_encoder.parameters(), lr=0.001)

iters = []
loss_arr, accuracy_arr, roc_auc_arr = [], [], []

# TRAIN LOOP
for epoch in range(TOTAL_EPOCHS):
    
    print(f"Start epoch - {epoch}")
    
    for iter, batch in enumerate(train_dl):
        xb, yb = batch[0], batch[1] 
        pred, loss = model(xb, yb, torch.einsum, torch.matmul, torch.bmm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            loss_train, acc_train, roc_auc_train= eval_model(model,
                                                    xb, yb, 
                                                    torch.einsum, 
                                                    torch.matmul, 
                                                    torch.bmm,
                                                    torch.nn.NLLLoss(),
                                                    torchmetrics.AUROC(task="multiclass", num_classes=2), 
                                                    torchmetrics.Accuracy(task="multiclass", num_classes=2))
            
            iters.append(epoch * MAX_ITERS + iter)
            loss_arr.append(loss_train.item())
            accuracy_arr.append(acc_train)
            #roc_auc_arr.append(roc_auc_train)
            
            print(f"Train epoch {epoch}, iteration {iter}: loss - {loss_train:.4f}, accuracy - {acc_train:.4f}") #, roc_auc - {roc_auc_train:.4f}")

        if iter >= MAX_ITERS:
            break
    
plt.plot(iters, loss_arr, color='y', label='Loss') 
plt.plot(iters, accuracy_arr, color='r', label='Accuracy')
#plt.plot(iters, roc_auc_arr, color='g', label='ROC_AUC')

plt.xlabel("Epoch") 
plt.ylabel("Magnitude") 
plt.title("Education")
plt.legend()
plt.grid()
plt.savefig(f"C:\\Users\\kruglovdy\\Desktop\\Finbert\\Education.png")
plt.show()

noise_arr = []
loss_arr, accuracy_arr, roc_auc_arr = [], [], []
        
# EVALUATION LOOP
for i in range(6):
 
    noise = 0.1*i
    loss_val_arr, acc_val_arr, roc_auc_val_arr = [], [], []
    ff = FuncFactory(noise)

    for iter, batch in enumerate(val_dl):
        
        xb_val, yb_val = batch[0], batch[1]
        loss_val, acc_val = eval_model(model, 
                                       xb_val, yb_val, 
                                       ff.einsum_with_noise, 
                                       ff.matmul_with_noise, 
                                       ff.bmm_with_noise,
                                       torch.nn.NLLLoss(),
                                       torchmetrics.AUROC(task="multiclass", num_classes=2), 
                                       torchmetrics.Accuracy(task="multiclass", num_classes=2))
        
        loss_val_arr.append(loss_val.item())
        acc_val_arr.append(acc_val)
        #roc_auc_val_arr.append(roc_auc_val.item())
        
        print(f"Validation step {iter}: loss - {loss_val:.4f}, accuracy - {acc_val:.4f}") #, roc_auc - {roc_auc_val:.4f}")
        
    noise_arr.append(noise)
    loss_arr.append(statistics.mean(loss_val_arr))
    accuracy_arr.append(statistics.mean(acc_val_arr))
    #roc_auc_arr.append(statistics.mean(roc_auc_val_arr))
     
plt.plot(noise_arr, loss_arr, color='y', label='Loss') 
plt.plot(noise_arr, accuracy_arr, color='r', label='Accuracy')
#plt.plot(noise_arr, roc_auc_arr, color='g', label='ROC_AUC')

plt.xlabel("Noise percent") 
plt.ylabel("Magnitude") 
plt.title("Matrix multiplication noise effect")
plt.legend()
plt.grid()
plt.savefig(f"C:\\Users\\kruglovdy\\Desktop\\Finbert\\Noise_on_inference.png")
plt.show()
