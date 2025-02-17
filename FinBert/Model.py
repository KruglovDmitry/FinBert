import numpy as np

import torch
import torchmetrics
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics.classification import accuracy
from transformers import BertConfig
from BERT import BertModel

from functools import partial
from ptls.nn import Head
from ptls.data_load.padded_batch import PaddedBatch
from torchmetrics import MeanMetric
from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from torch.nn import BCELoss
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryROC

class PretrainModule(pl.LightningModule):
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_size: int,
                 loss_temperature: float,
                 total_steps: int,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = False,
                 num_attention_heads: int = 16,
                 intermediate_size: int = 128,
                 num_hidden_layers: int = 2,
                 attention_window: int = 16,
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings: int = 4000,
                 replace_proba: float = 0.1,
                 neg_count: int = 1,
                 log_logits: bool = False,
                 encode_seq = False,
                 enable_optica=False):
        """

        Parameters
        ----------
        encoder:
            Module for transform dict with feature sequences to sequence of transaction representations
        hidden_size:
            Output size of `encoder`. Hidden size of internal transformer representation
        loss_temperature:
             temperature parameter of `QuerySoftmaxLoss`
        total_steps:
            total_steps expected in OneCycle lr scheduler
        max_lr:
            max_lr of OneCycle lr scheduler
        weight_decay:
            weight_decay of Adam optimizer
        pct_start:
            % of total_steps when lr increase
        norm_predict:
            use l2 norm for transformer output or not
        num_attention_heads:
            parameter for Longformer
        intermediate_size:
            parameter for Longformer
        num_hidden_layers:
            parameter for Longformer
        attention_window:
            parameter for Longformer
        max_position_embeddings:
            parameter for Longformer
        replace_proba:
            probability of masking transaction embedding
        neg_count:
            negative count for `QuerySoftmaxLoss`
        log_logits:
            if true than logits histogram will be logged. May be useful for `loss_temperature` tuning
        encode_seq:
            if true then outputs zero element of transformer i.e. encodes whole sequence. Else returns all outputs of transformer except 0th.
        """

        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'head', 'loss', 'metric_list', 'optimizer_partial', 'lr_scheduler_partial'], logger=False)
        self.softmax = torch.nn.Softmax()
        self.loss = torch.nn.BCELoss()
        self.preds, self.target = None, None
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.roc_auc = torchmetrics.AUROC(task="multiclass", num_classes=2)
        self.roc_metric = BinaryROC()
        
        self.optimizer_partial = partial(torch.optim.Adam, lr=0.001),
        self.lr_scheduler_partial = partial(torch.optim.lr_scheduler.StepLR, step_size=2000, gamma=1)
        
        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        
        self.encoder = encoder
        
        self.transf = BertModel(
            config=BertConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=4,
                max_position_embeddings=self.hparams.max_position_embeddings,
                attention_window=attention_window,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob
            ),
            add_pooling_layer=False,
            enable_optica=enable_optica,
        )

        self.head = Head(
            input_size=64,
            hidden_layers_sizes=[32, 8],
            drop_probs=[0.1, 0],
            use_batch_norm=True,
            objective='classification',
            num_classes=2,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.max_lr, weight_decay=self.hparams.weight_decay,)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
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
        return [optimizer], [scheduler]

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

        if self.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
        
        out_head = self.head(out[:, 0])
        
        return out_head

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(self.softmax(y_hat)[:,1].to('cpu'), y.type(torch.FloatTensor))
        accuracy = self.accuracy(y_hat, y)
        roc_auc = self.roc_auc(y_hat, y)

        self.log('train_loss', loss) 
        self.log('train_accuracy', accuracy)
        self.log('train_roc_auc', roc_auc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(self.softmax(y_hat)[:,1].to('cpu'), y.type(torch.FloatTensor))
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
        fig_.savefig('/wd/finbert/roc_curve_bert_bce.png')


