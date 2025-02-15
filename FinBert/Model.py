import numpy as np

import torch
import torchmetrics
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics.classification import accuracy
from torchmetrics.classification import BinaryROC
from transformers import LlamaConfig 
from Llama import LlamaModel
from ptls.nn import TrxEncoder, PBLinear, PBL2Norm, PBLayerNorm, PBDropout

from functools import partial
from ptls.nn import Head
from ptls.data_load.padded_batch import PaddedBatch
from torchmetrics import MeanMetric
from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from torch.nn import BCELoss
from torchmetrics import MetricCollection

class PretrainModule(pl.LightningModule):
    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 total_steps: int,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 hidden_size: int = 64,
                 intermediate_size: int = 128,
                 num_hidden_layers: int = 2,
                 num_attention_heads: int = 16,
                 max_position_embeddings: int = 4000,
                 rms_norm_eps: float = 1e-6,
                 initializer_range: float = 0.02,
                 use_cache: bool = True,
                 tie_word_embeddings: bool = False,
                 enable_optica=False,):

        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'head', 'loss', 'metric_list', 'optimizer_partial', 'lr_scheduler_partial'], logger=False)

        self.loss = torch.nn.NLLLoss()
        self.preds, self.target = None, None

        self.roc_metric = BinaryROC()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.roc_auc = torchmetrics.AUROC(task="multiclass", num_classes=2)
        
        self.optimizer_partial = partial(torch.optim.Adam, lr=0.001),
        self.lr_scheduler_partial = partial(torch.optim.lr_scheduler.StepLR, step_size=2000, gamma=1)
        
        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        
        self.encoder = torch.nn.Sequential(
            trx_encoder,
            PBLinear(trx_encoder.output_size, 64),
            PBDropout(0.2))
        
        self.transf = LlamaModel(
            config=LlamaConfig(
                vocab_size=4,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                initializer_range=initializer_range,
                use_cache=use_cache,
                tie_word_embeddings=tie_word_embeddings,),
            enable_optica=enable_optica,)

        self.head = Head(
            input_size=64,
            hidden_layers_sizes=[32, 8],
            drop_probs=[0.1, 0],
            use_batch_norm=True,
            objective='classification',
            num_classes=2,
        )
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams.max_lr,
                                 weight_decay=self.hparams.weight_decay,
                                 )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]

    def forward(self, z: PaddedBatch):
        z = self.encoder(z)

        B, T, H = z.payload.size()
        device = z.payload.device

        if self.training:
            start_pos = np.random.randint(0, self.hparams.max_position_embeddings - T - 1, 1)[0]
        else:
            start_pos = 0

        inputs_embeds = z.payload
        attention_mask = z.seq_len_mask.float()

        inputs_embeds = torch.cat([
            self.token_cls.expand(inputs_embeds.size(0), 1, H),
            inputs_embeds,
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones(inputs_embeds.size(0), 1, device=device),
            attention_mask,
        ], dim=1)
        position_ids = torch.arange(T + 1, device=z.device).view(1, -1).expand(B, T + 1) + start_pos
        global_attention_mask = torch.cat([
            torch.ones(inputs_embeds.size(0), 1, device=device),
            torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1) - 1, device=device),
        ], dim=1)

        out = self.transf(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state
        
        out_head = self.head(out[:, 0])

        return out_head

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        roc_auc = self.roc_auc(y_hat, y)

        self.log('train_loss', loss) 
        self.log('train_accuracy', accuracy)
        self.log('train_roc_auc', roc_auc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        roc_auc = self.roc_auc(y_hat, y)
        
        self.log('val_loss', loss) 
        self.log('val_accuracy', accuracy)
        self.log('val_roc_auc', roc_auc)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        self.preds = y_hat if self.preds is None else torch.cat((self.preds, y_hat))
        self.target = y if self.target is None else torch.cat((self.target, y))

    def on_test_end(self):
        print("ROC_AUC score - ", self.roc_auc(self.preds, self.target).item())
        self.roc_metric.update(self.preds[:, 1], self.target)
        fig_, ax_ = self.roc_metric.plot()
        fig_.savefig('/wd/finbert/roc_curve_optic_fit.png')

