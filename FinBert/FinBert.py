import os
import sys
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
from ptls.nn import TrxEncoder, Head, PBLinear, PBL2Norm, PBLayerNorm, PBDropout
from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing import SeqLenFilter, FeatureFilter, ToTorch
from ptls.frames.supervised.seq_to_target_dataset import SeqToTargetIterableDataset
from ptls.data_load import IterableChain
from ptls.frames import PtlsDataModule
from Model import MLMCPCPretrainModule
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

# logger = pl.loggers.TensorBoardLogger(
#     save_dir='.',
#     name='lightning_logs',
#     version='_16_heads_2_layers_no_dropout_at_all_no_norm_emb_drop_0.3_PBDrop_0.2_no_transf_norm_drop_in_head_0.1')

# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     dirpath="./ckpts/_16_heads_2_layers_no_dropout_at_all_no_norm_emb_drop_0.3_PBDrop_0.2_no_transf_norm_drop_in_head_0.1/", 
#     save_top_k=40, 
#     mode='max', 
#     monitor="valid/MulticlassAUROC")

MAX_EPOCHS = 10
LIMIT_TRAIN_BATCHES = 10
LIMIT_VAL_BATCHES = 1
MAX_ITERS = 5000
EVAL_INTERVAL = 200

noise_arr = []
loss_arr, accuracy_arr = [], []

for i in range(6):
 
    noise = 0.1*i
    loss_val_arr, acc_val_arr = [], []
    optimizer = torch.optim.AdamW(seq_encoder.parameters(), lr=0.001)
    ff = FuncFactory(noise)

    train_dl = finetune_dm.train_dl()
    for iter, batch in enumerate(train_dl):
        xb, yb = batch[0], batch[1] 
        pred, loss = seq_encoder(xb, ff.einsum, ff.matmul)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            loss_train, acc_train = seq_encoder.eval_model(xb, yb)
            print(f"Train step {iter}: loss {loss_train:.4f}, accuracy {acc_train:.4f}")
        
        if iter >= MAX_ITERS:
            break
            
    val_dl = finetune_dm.val_dl()
    for iter, batch in enumerate(val_dl):
        xb_val, yb_val = batch[0], batch[1]
        loss_val, acc_val = seq_encoder.eval_model(xb_val, yb_val)
        loss_val_arr.append(loss_val)
        acc_val_arr.append(acc_val)
        print(f"Validation step {iter}: loss {loss_val:.4f}, accuracy {acc_val:.4f}")
        
    noise_arr.append(noise)
    loss_arr.append(statistics.mean(loss_val_arr))
    accuracy_arr.append(statistics.mean(acc_val_arr))
    
    # downstream_model = SequenceToTarget(
    #     seq_encoder=seq_encoder,
    #     head=Head(
    #         input_size=64,
    #         hidden_layers_sizes=[32, 8],
    #         drop_probs=[0.1, 0],
    #         use_batch_norm=True,
    #         objective='classification',
    #         num_classes=2,
    #     ),
    #     loss=torch.nn.NLLLoss(),
    #     metric_list=[torchmetrics.AUROC(task="multiclass", num_classes=2), torchmetrics.Accuracy(task="multiclass", num_classes=2)],
    #     pretrained_lr=0.001,
    #     optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    #     lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=2000, gamma=1),)

    # trainer = pl.Trainer(
    #     max_epochs=MAX_EPOCHS, 
    #     accelerator="auto", 
    #     enable_progress_bar=True, 
    #     limit_train_batches=LIMIT_TRAIN_BATCHES,
    #     limit_val_batches=LIMIT_VAL_BATCHES, 
    #     callbacks=[checkpoint_callback],
    #     logger=logger)
    
    # print(f'logger.version = {trainer.logger.version}')
    # trainer.fit(downstream_model, finetune_dm)
    
    # print(trainer.logged_metrics)
    # loss_arr.append(trainer.logged_metrics['loss'].item())
    # roc_auc_arr.append(trainer.logged_metrics['valid/MulticlassAUROC'].item())
    # accuracy_arr.append(trainer.logged_metrics['valid/MulticlassAccuracy'].item())
    # noise_arr.append(noise_percent)
 
plt.plot(noise_arr, loss_arr, color='y', label='Loss') 
plt.plot(noise_arr, accuracy_arr, color='r', label='Accuracy')

plt.xlabel("Noise percent") 
plt.ylabel("Magnitude") 
plt.title("Matrix multiplication noise effect")
plt.legend()
plt.grid()
#plt.savefig(f"C:\\Users\\kruglovdy\\Desktop\\Finbert\\Result.png")
plt.show()

#df = pd.DataFrame(np.array([noise_arr, loss_arr, accuracy_arr]))
#df.to_excel('C:\\Users\\kruglovdy\\Desktop\\Finbert\\Result.xlsx')

# ��� batch_size = 128 ���������� �������� � ����� ����� �� ���� ������ ����� 23 205 ��� ���� ����� �������� 5 ����� 45 �����
# ������ tensorboard �� cmd - tensorboard --logdir=C:\Users\kruglovdy\source\repos\FinBert\FinBert\lightning_logs\_16_heads_2_layers_no_dropout_at_all_no_norm_emb_drop_0.3_PBDrop_0.2_no_transf_norm_drop_in_head_0.1
# matmul, einsum 
# start 0..9 percents with 10 epochs - 14.15 end 16.11 (2 hours)
