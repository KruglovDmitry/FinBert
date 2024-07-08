import os
import sys
import numpy as np
import random
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
    spatial_dropout=False
)

trx_encoder = TrxEncoder(**trx_encoder_params)

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
    encode_seq=True
)

downstream_model = SequenceToTarget(
    seq_encoder=seq_encoder,
    head=Head(
        input_size=64,
        hidden_layers_sizes=[32, 8],
        drop_probs=[0.1, 0],
        use_batch_norm=True,
        objective='classification',
        num_classes=2,
    ),
    loss=torch.nn.NLLLoss(),
    metric_list=[torchmetrics.AUROC(task="multiclass", num_classes=2), torchmetrics.Accuracy(task="multiclass", num_classes=2)],
    pretrained_lr=0.001,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=2000, gamma=1),
)


train_pq_files = ParquetFiles('..\\finbert_data\\train.parquet\\')
valid_pq_files = ParquetFiles('..\\finbert_data\\valid.parquet\\')

train_dataset = ParquetDataset(data_files=train_pq_files.data_files, shuffle_files=True)
valid_dataset = ParquetDataset(data_files=valid_pq_files.data_files, shuffle_files=True)

finetune_dm = PtlsDataModule(
    train_data=SeqToTargetIterableDataset(train_dataset, target_col_name='flag'),
    valid_data=SeqToTargetIterableDataset(valid_dataset, target_col_name='flag'),
    train_num_workers=0, #20
    train_batch_size=1024, #1024,
    valid_batch_size=1024, #1024,
)

logger = pl.loggers.TensorBoardLogger(
                save_dir='.',
                name='lightning_logs',
                version='_16_heads_2_layers_no_dropout_at_all_no_norm_emb_drop_0.3_PBDrop_0.2_no_transf_norm_drop_in_head_0.1'
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./ckpts/_16_heads_2_layers_no_dropout_at_all_no_norm_emb_drop_0.3_PBDrop_0.2_no_transf_norm_drop_in_head_0.1/", save_top_k=40, mode='max', monitor="valid/MulticlassAUROC")

trainer_ft = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    enable_progress_bar=True,
    limit_train_batches=10,
    limit_val_batches=1,
    callbacks=[checkpoint_callback],
    logger=logger
)

print(f'logger.version = {trainer_ft.logger.version}')
trainer_ft.fit(downstream_model, finetune_dm)
print(trainer_ft.logged_metrics)

# ��� batch_size = 128 ���������� �������� � ����� ����� �� ���� ������ ����� 23 205 ��� ���� ����� �������� 5 ����� 45 �����
# ������ tensorboard �� cmd - tensorboard --logdir=C:\Users\kruglovdy\source\repos\FinBert\FinBert\lightning_logs\_16_heads_2_layers_no_dropout_at_all_no_norm_emb_drop_0.3_PBDrop_0.2_no_transf_norm_drop_in_head_0.1