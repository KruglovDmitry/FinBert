import os
import torch
import pickle
import pytorch_lightning as pl
from ptls.nn import PBLinear, PBL2Norm, PBLayerNorm, PBDropout

from torch.utils.data import DataLoader, TensorDataset
from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.frames.supervised.seq_to_target_dataset import SeqToTargetIterableDataset
from ptls.frames import PtlsDataModule

from Encoder import TrxEncoder
from Model import PretrainModule

print("Pythorch version - ", torch.__version__)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.multiprocessing.set_sharing_strategy('file_system')

# load preprocessed data:
with open('..\\finbert_data_num\\preproc_data', 'rb') as h:
    preproc_data = pickle.load(h)
    numeric_values = preproc_data['numeric_values']
    embeddings = preproc_data['embeddings']

test_pq_files  = ParquetFiles('..\\finbert_data_num\\test.parquet\\')
train_pq_files = ParquetFiles('..\\finbert_data_num\\train.parquet\\')
valid_pq_files = ParquetFiles('..\\finbert_data_num\\valid.parquet\\')

test_dataset  = ParquetDataset(data_files=test_pq_files.data_files, shuffle_files=True)
train_dataset = ParquetDataset(data_files=train_pq_files.data_files, shuffle_files=True)
valid_dataset = ParquetDataset(data_files=valid_pq_files.data_files, shuffle_files=True)

finetune_dm = PtlsDataModule(
    test_data=SeqToTargetIterableDataset(test_dataset, target_col_name='flag'),
    train_data=SeqToTargetIterableDataset(train_dataset, target_col_name='flag'),
    valid_data=SeqToTargetIterableDataset(valid_dataset, target_col_name='flag'),
    train_num_workers=0, #20
    test_batch_size=24,
    train_batch_size=1024,
    valid_batch_size=1024,)

logger = pl.loggers.TensorBoardLogger(save_dir='.', name='lightning_logs', version='BERT')
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./ckpts/BERT/", save_top_k=40, mode='max', monitor="val_roc_auc")

### СОЗДАНИЕ МОДЕЛИ ###

trx_encoder_params = dict(
    embeddings_noise=0,
    numeric_values=numeric_values,
    embeddings=embeddings,
    emb_dropout=0.3,
    spatial_dropout=False)

trx_encoder = TrxEncoder(**trx_encoder_params)
encoder = torch.nn.Sequential(
            trx_encoder,
            PBLinear(trx_encoder.output_size, 128),
            PBDropout(0.2))

model = PretrainModule(
        encoder=encoder,
        hidden_size=128,
        loss_temperature=20.0,
        total_steps=30000,
        replace_proba=0.1,
        neg_count=64,
        log_logits=False,
        encode_seq=True,)

trainer = pl.Trainer(
    max_epochs=100, 
    accelerator="auto", 
    enable_progress_bar=True, 
    #limit_train_batches=10,
    #limit_val_batches=1, 
    callbacks=[checkpoint_callback],
    logger=logger)
    
print(f'logger.version = {trainer.logger.version}')
trainer.fit(model, finetune_dm)

print(trainer.logged_metrics)
trainer.test(model, dataloaders=finetune_dm)

### ЗАГРУЗКА МОДЕЛИ ###

loaded_model = PretrainModule.load_from_checkpoint(
        './ckpts/BERT/epoch=4-step=50.ckpt',
        encoder=encoder,
        hidden_size=128,
        loss_temperature=20.0,
        total_steps=30000,
        replace_proba=0.1,
        neg_count=64,
        log_logits=False,
        encode_seq=True,
        enable_optica=True)

trainer.test(loaded_model, dataloaders=finetune_dm)