from  pytorch_lightning.core import LightningDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import  WandbLogger, TensorBoardLogger
from pytorch_lightning.loggers.logger import  DummyLogger
from pytorch_lightning import Trainer
from torchmetrics import MeanMetric
from typing import Optional
from collections import OrderedDict
import os 
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from eend.pytorch_backend.diarization_dataset import KaldiDiarizationDataset, my_collate
from eend.pytorch_backend.loss import  speakers_loss,batch_pit_n_speaker_loss, DiarizationErrorPyannoteProcess, DiarizationErrorPyannote, DiarizationErrorPyannoteProcessMetric
from eend.pytorch_backend import utils
from eend.pytorch_backend.models import TransformerModel, TransformerEDADiarization, TitanetEDADiarization, NoamScheduler
from eend import feature
import wandb
import time
import json
class KaldiDiarizationLightningData(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.collate_fn = my_collate

    def setup(self, stage: Optional[str] = None):
        config = self.config
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            print("Loading train dataset")
            self.train_dataset = KaldiDiarizationDataset(
            data_dir=config.train_data_dir,
            chunk_size=config.num_frames,
            context_size=config.context_size,
            input_transform=config.input_transform,
            frame_size=config.frame_size,
            frame_shift=config.frame_shift,
            subsampling=config.subsampling,
            rate=config.sampling_rate,
            use_last_samples=True,
            label_delay=config.label_delay,
            n_speakers=config.num_speakers,
            model_sr=config.model_sr,
            raw_wav=config.raw_wav,
            )
            print("Loading val dataset")
            self.val_dataset =  KaldiDiarizationDataset(
                                data_dir=config.valid_data_dir,
                                chunk_size=config.num_frames,
                                context_size=config.context_size,
                                input_transform=config.input_transform,
                                frame_size=config.frame_size,
                                frame_shift=config.frame_shift,
                                subsampling=config.subsampling,
                                rate=config.sampling_rate,
                                use_last_samples=True,
                                label_delay=config.label_delay,
                                n_speakers=config.num_speakers,
                                model_sr=config.model_sr,
                                raw_wav=config.raw_wav,
                                )
       

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:

            self.test_dataset = KaldiDiarizationDataset(
                                data_dir=config.test_data_dir,
                                chunk_size=config.test_num_frames,
                                context_size=config.context_size,
                                input_transform=config.input_transform,
                                frame_size=config.frame_size,
                                frame_shift=config.frame_shift,
                                subsampling=config.subsampling,
                                rate=config.sampling_rate,
                                use_last_samples=True,
                                label_delay=config.label_delay,
                                n_speakers=config.num_speakers,
                                model_sr=config.model_sr,
                                raw_wav=config.raw_wav,
                                )

    def train_dataloader(self):
        train_dataset = self.train_dataset

        # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        sampler = utils.DistributedSubsetSampler(train_dataset, part=0.2)
        # sampler = utils.DistributedSubsetSampler(train_dataset, part=1)
        shuffle = False 
        num_workers = self.config.num_workers
        dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=self.config.batchsize, shuffle=shuffle, collate_fn=self.collate_fn,
                        num_workers=num_workers, pin_memory=True, sampler=sampler)
        return dataloader

    def create_val_loader(self, dataset):

        shuffle = False
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)

        num_workers = self.config.num_workers
        val_dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=self.config.batchsize, shuffle=shuffle, collate_fn=self.collate_fn,
                        num_workers=num_workers, pin_memory=True, sampler=sampler)
        return val_dataloader
        
    def val_dataloader(self):
        # option to more than 1 val dataloader
        loaders_list = []
        val_dataset = self.val_dataset
        dataloader_val = self.create_val_loader(val_dataset)
        loaders_list.append(dataloader_val)

        return loaders_list

    def test_dataloader(self):
        test_dataset = self.test_dataset
   
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        num_workers = self.config.num_workers
        dataloader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=4, shuffle=False, collate_fn=self.collate_fn,
                        num_workers=num_workers, pin_memory=True, sampler=sampler)
        return dataloader

class TransformerEDADiarizationLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config    
        self.config["speaker_threshold"] = config.get("speaker_threshold", 0.5)
        self.config["max_num_speakers"] = config.get("max_num_speakers", -1)
        if config["input_transform"] == 'log':  #
            n_fft = 1 << (self.config.frame_size-1).bit_length()
            in_size = 1 + n_fft/2
        else:
            in_size = feature.get_nfeatures(config["input_transform"]) * (config["context_size"] * 2 + 1)
        self.model = TransformerEDADiarization( in_size=in_size,
                                        n_units=self.config["hidden_size"],
                                        n_heads=self.config["transformer_encoder_n_heads"],
                                        n_layers=self.config["transformer_encoder_n_layers"],
                                        dropout=self.config["transformer_encoder_dropout"],
                                        attractor_decoder_dropout=self.config["attractor_decoder_dropout"],
                                        attractor_encoder_dropout=self.config["attractor_encoder_dropout"],
                                        has_pos=False, shuffle=self.config["shuffle"])
                

        self.train_loss_tracker = MeanMetric()
        self.pit_los_tracker = MeanMetric()
        self.speakers_loss_tracker = MeanMetric()
        self.val_loss_tracker = MeanMetric()
        self.val_pit_los_tracker = MeanMetric()
        self.val_speakers_loss_tracker = MeanMetric()
        self.pyannote_stats = DiarizationErrorPyannoteProcessMetric(collar=self.config["collar"],skip_overlap=self.config["skip_overlap"],
                                        window=(self.config["frame_shift"]/self.config["sampling_rate"])*self.config["subsampling"])
  
        self.save_hyperparameters(config)
        self.last_time = time.time()

    def update_max_speakers(self, max_n_speakers):
        self.config["max_num_speakers"] = max_n_speakers

    def forward(self, y_tensor, n_speakers):
        # in lightning, forward defines the prediction/inference actions
        logits, attractors_prob = self.model(y_tensor, n_speakers=n_speakers) # TODO- implement forward to infererence
        return logits, attractors_prob 

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def freeze(self, freeze):
        self.model.freeze(freeze)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward

        device = self.device
        y, t = batch
        n_speakers = [ti.shape[1] for ti in t] 
        n_speakers_active = [(torch.sum(ti,axis=0) > 0).sum().item() for ti in t] 
        lens = [ti.shape[0] for ti in t]
        y_tensor = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)

        output = self.model(y_tensor, n_speakers=n_speakers)

        logits, attractors_prob = output
        
        mask = utils.build_mask_by_len(lens, device=device)

        pit_loss, label = batch_pit_n_speaker_loss(logits, t, n_speakers,mask)

        total_loss = pit_loss
        self.pit_los_tracker(pit_loss)
        if self.config["attractor_loss_ratio"] > 0 :

            batch_speakers_loss = speakers_loss(attractors_prob, n_speakers_active, device)
            self.speakers_loss_tracker(batch_speakers_loss)
            total_loss += self.config["attractor_loss_ratio"] * batch_speakers_loss

        self.train_loss_tracker(total_loss)

        return total_loss.mean()


    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train/loss", self.train_loss_tracker.compute(), on_step=False,on_epoch=True, prog_bar=True)
        self.log("train/pit_loss", self.pit_los_tracker.compute(), on_step=False,on_epoch=True, prog_bar=False)
        self.log("train/batch_speakers_loss", self.speakers_loss_tracker.compute(), on_step=False,on_epoch=True, prog_bar=False)

        self.train_loss_tracker.reset()
        self.pit_los_tracker.reset()
        self.speakers_loss_tracker.reset()

        
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")


    def eval_step(self, batch, batch_idx, step_type):
        device = self.device
        y, t = batch
        n_speakers = [ti.shape[1] for ti in t] 
        n_speakers_active = [(torch.sum(ti,axis=0) > 0).sum().item() for ti in t] 
        lens = [ti.shape[0] for ti in t]
        y_tensor = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)

        if step_type == "test":
            max_n_speakers = self.config["max_num_speakers"] if self.config["max_num_speakers"] > 0 else max(n_speakers)
            logits, attractors_prob = self.model.estimate(y_tensor, n_speakers=max_n_speakers, threshold=self.config["speaker_threshold"])
            n_speakers_tensor = torch.sum((attractors_prob.squeeze(2) > self.config["speaker_threshold"]),axis=1)
            if self.config["max_num_speakers"] > 0: # if max_num_speakers is set, we need to update the n_speakers acoording to attractors_prob 
                n_speakers_der = n_speakers_tensor.cpu().numpy().tolist()
                n_speakers_der = [min(spk_n, max_n_speakers) for spk_n in n_speakers_der ]
            else:
                n_speakers_der = n_speakers
        else:
            logits, attractors_prob = self.model(y_tensor, n_speakers=n_speakers)
            n_speakers_der = n_speakers
        mask = utils.build_mask_by_len(lens, device=device)

        pit_loss, label = batch_pit_n_speaker_loss(logits, t, n_speakers,mask)
        total_loss = pit_loss
        self.val_pit_los_tracker(pit_loss)
        if self.config["attractor_loss_ratio"] > 0 :
            batch_speakers_loss = speakers_loss(attractors_prob, n_speakers_active, device)
            self.val_speakers_loss_tracker(batch_speakers_loss)
            total_loss += self.config["attractor_loss_ratio"] * batch_speakers_loss

        self.val_loss_tracker(total_loss)

        batch_states = self.pyannote_stats(logits,label, lens, n_speakers_der)

        self.log(f'{step_type}/loss', self.val_loss_tracker, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{step_type}/pit_loss', self.val_pit_los_tracker, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{step_type}/speakers_loss', self.val_speakers_loss_tracker, on_step=False, on_epoch=True, prog_bar=False)
        

        return OrderedDict({f'{step_type}_states': batch_states})

    def validation_epoch_end(self, outputs): 
        return self.eval_step_end(outputs, "val")


    def test_epoch_end(self, outputs): 
        return self.eval_step_end(outputs, "test")


    def eval_step_end(self, outputs, step_type):

        all_states_tensor = self.pyannote_stats.compute()
        for key, value in zip(["DER", "confusion", "missed detection","false alarm","correct","total"],all_states_tensor):
            prog_bar = True if key == "DER" or key=="total" else False
            self.log(f'{step_type}/{key}', value, on_step=False, on_epoch=True, prog_bar=prog_bar)
            
        self.pyannote_stats.reset()
        self.val_loss_tracker.reset()
        self.val_pit_los_tracker.reset()
        self.val_speakers_loss_tracker.reset()

    def configure_optimizers(self):
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config["lr"])
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config["lr"])
        elif self.config['optimizer'] == 'noam':
            # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config["lr"], betas=(0.9, 0.98), eps=1e-9)
        else:
            raise ValueError(self.config['optimizer'])

        # For noam, we use noam scheduler
        if self.config["optimizer"] == 'noam':
            scheduler = NoamScheduler(optimizer,
                                    self.config["hidden_size"],
                                    warmup_steps=self.config["noam_warmup_steps"])
        
        else:
            scheduler = None
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config["stop_measure"],
                },
                }

class TitanetEDADiarizationLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config    
        self.config["speaker_threshold"] = config.get("speaker_threshold", 0.5)
        self.config["max_num_speakers"] = config.get("max_num_speakers", -1)
        self.model = TitanetEDADiarization( 
                                        n_units=self.config["hidden_size"],
                                        context=self.config["context_size"],
                                        attractor_decoder_dropout=self.config["attractor_decoder_dropout"],
                                        attractor_encoder_dropout=self.config["attractor_encoder_dropout"],
                                        shuffle=self.config["shuffle"],freeze_encoder=self.config["freeze_encoder"]) 

        self.train_loss_tracker = MeanMetric()
        self.pit_los_tracker = MeanMetric()
        self.speakers_loss_tracker = MeanMetric()
        self.val_loss_tracker = MeanMetric()
        self.val_pit_los_tracker = MeanMetric()
        self.val_speakers_loss_tracker = MeanMetric()
        self.pyannote_stats = DiarizationErrorPyannoteProcessMetric(collar=self.config["collar"],skip_overlap=self.config["skip_overlap"],
                                        window=(self.config["frame_shift"]/self.config["sampling_rate"])*self.config["subsampling"])
  
        self.save_hyperparameters(config)
        self.last_time = time.time()

    def update_max_speakers(self, max_n_speakers):
        self.config["max_num_speakers"] = max_n_speakers

    def forward(self, y_tensor, n_speakers):
        # in lightning, forward defines the prediction/inference actions
        logits, attractors_prob = self.model(y_tensor, n_speakers=n_speakers) # TODO- implement forward to infererence
        return logits, attractors_prob 

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def freeze(self, freeze):
        self.model.freeze(freeze)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward

        device = self.device
        y, t = batch
        n_speakers = [ti.shape[1] for ti in t] 
        n_speakers_active = [(torch.sum(ti,axis=0) > 0).sum().item() for ti in t] 
        lens = [ti.shape[0] for ti in t]
        y_tensor = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)
        xlens = [yi.shape[0] for yi in y]
        output = self.model(y_tensor, xlens=xlens, n_speakers=n_speakers)

        logits, attractors_prob = output
        
        mask = utils.build_mask_by_len(lens, device=device)

        pit_loss, label = batch_pit_n_speaker_loss(logits, t, n_speakers,mask)

        total_loss = pit_loss
        self.pit_los_tracker(pit_loss)
        if self.config["attractor_loss_ratio"] > 0 :

            batch_speakers_loss = speakers_loss(attractors_prob, n_speakers_active, device)
            self.speakers_loss_tracker(batch_speakers_loss)
            total_loss += self.config["attractor_loss_ratio"] * batch_speakers_loss

        self.train_loss_tracker(total_loss)

        return total_loss.mean()


    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train/loss", self.train_loss_tracker.compute(), on_step=False,on_epoch=True, prog_bar=True)
        self.log("train/pit_loss", self.pit_los_tracker.compute(), on_step=False,on_epoch=True, prog_bar=False)
        self.log("train/batch_speakers_loss", self.speakers_loss_tracker.compute(), on_step=False,on_epoch=True, prog_bar=False)

        self.train_loss_tracker.reset()
        self.pit_los_tracker.reset()
        self.speakers_loss_tracker.reset()

        
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")


    def eval_step(self, batch, batch_idx, step_type):
        device = self.device
        y, t = batch
        n_speakers = [ti.shape[1] for ti in t] 
        n_speakers_active = [(torch.sum(ti,axis=0) > 0).sum().item() for ti in t] 
        lens = [ti.shape[0] for ti in t]
        xlens = [yi.shape[0] for yi in y]
        y_tensor = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)

        if step_type == "test":
            max_n_speakers = self.config["max_num_speakers"] if self.config["max_num_speakers"] > 0 else max(n_speakers)
            logits, attractors_prob = self.model.estimate(y_tensor, xlens=xlens, n_speakers=max_n_speakers, threshold=self.config["speaker_threshold"])
            n_speakers_tensor = torch.sum((attractors_prob.squeeze(2) > self.config["speaker_threshold"]),axis=1)
            if self.config["max_num_speakers"] > 0: # if max_num_speakers is set, we need to update the n_speakers acoording to attractors_prob 
                n_speakers_der = n_speakers_tensor.cpu().numpy().tolist()
                n_speakers_der = [min(spk_n, max_n_speakers) for spk_n in n_speakers_der ]
            else:
                n_speakers_der = n_speakers
        else:
            logits, attractors_prob = self.model(y_tensor, xlens=xlens, n_speakers=n_speakers)
            n_speakers_der = n_speakers
        mask = utils.build_mask_by_len(lens, device=device)

        pit_loss, label = batch_pit_n_speaker_loss(logits, t, n_speakers,mask)
        total_loss = pit_loss
        self.val_pit_los_tracker(pit_loss)
        if self.config["attractor_loss_ratio"] > 0 :
            batch_speakers_loss = speakers_loss(attractors_prob, n_speakers_active, device)
            self.val_speakers_loss_tracker(batch_speakers_loss)
            total_loss += self.config["attractor_loss_ratio"] * batch_speakers_loss

        self.val_loss_tracker(total_loss)

        batch_states = self.pyannote_stats(logits,label, lens, n_speakers_der)

        self.log(f'{step_type}/loss', self.val_loss_tracker, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{step_type}/pit_loss', self.val_pit_los_tracker, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{step_type}/speakers_loss', self.val_speakers_loss_tracker, on_step=False, on_epoch=True, prog_bar=False)
        

        return OrderedDict({f'{step_type}_states': batch_states})

    def validation_epoch_end(self, outputs): 
        return self.eval_step_end(outputs, "val")


    def test_epoch_end(self, outputs): 
        return self.eval_step_end(outputs, "test")


    def eval_step_end(self, outputs, step_type):

        all_states_tensor = self.pyannote_stats.compute()
        for key, value in zip(["DER", "confusion", "missed detection","false alarm","correct","total"],all_states_tensor):
            prog_bar = True if key == "DER" or key=="total" else False
            self.log(f'{step_type}/{key}', value, on_step=False, on_epoch=True, prog_bar=prog_bar)
            
        self.pyannote_stats.reset()
        self.val_loss_tracker.reset()
        self.val_pit_los_tracker.reset()
        self.val_speakers_loss_tracker.reset()

    def configure_optimizers(self):
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config["lr"])
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config["lr"])
        elif self.config['optimizer'] == 'noam':
            # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config["lr"], betas=(0.9, 0.98), eps=1e-9)
        else:
            raise ValueError(self.config['optimizer'])

        # For noam, we use noam scheduler
        if self.config["optimizer"] == 'noam':
            scheduler = NoamScheduler(optimizer,
                                    self.config["hidden_size"],
                                    warmup_steps=self.config["noam_warmup_steps"])
        
        else:
            scheduler = None
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config["stop_measure"],
                },
                }



def train(hparams):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """

    # Logger settings====================================================
    if not os.path.exists(hparams.model_save_dir):
        os.makedirs(hparams.model_save_dir)
    np.random.seed(hparams.seed)
    os.environ['PYTORCH_SEED'] = str(hparams.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    my_data_moudule = KaldiDiarizationLightningData(hparams)
    config = vars(hparams)
    if hparams.initmodel:
        print(f"Load model from {hparams.initmodel}")
        if hparams.model_type == 'Transformer':
            raise NotImplementedError # TODO: implement reg transformer for pytorch lightning
        elif hparams.model_type == 'EDATransformer':
            if hparams.initmodel.split("/")[-1].split(".")[-1] == "ckpt":
                model = TransformerEDADiarizationLightning.load_from_checkpoint(hparams.initmodel)
                model.update_max_speakers(hparams.max_num_speakers)
            else:
                model = TransformerEDADiarizationLightning(config)
                model.model.load_state_dict(torch.load(hparams.initmodel))
        elif hparams.model_type == 'TitanetEDA':
            if hparams.initmodel.split("/")[-1].split(".")[-1] == "ckpt":
                model = TitanetEDADiarizationLightning.load_from_checkpoint(hparams.initmodel)
                model.update_max_speakers(hparams.max_num_speakers)
            else:
                model = TitanetEDADiarizationLightning(config)
                model.model.load_state_dict(torch.load(hparams.initmodel))
        else:
            raise ValueError('Possible model_type is "Transformer" or "EDATransformer"')
    
    else:
        if hparams.model_type == 'Transformer':
            raise NotImplementedError # TODO: implement reg transformer for pytorch lightning
        elif hparams.model_type == 'EDATransformer':
            model = TransformerEDADiarizationLightning(config)
        elif hparams.model_type == 'TitanetEDA':
            model = TitanetEDADiarizationLightning(config)
        else:
            raise ValueError('Possible model_type is "Transformer" or "EDATransformer"')
    

    # training
    early_stop = EarlyStopping(
    monitor=hparams.stop_measure,
    patience=hparams.patience,
    verbose=True,
    mode=hparams.stop_measure_type
)
    loggers_list = []
 
    if hparams.wandb:
        if hparams.wandb_id is None:
            resume = "allow"
        else:
            resume = "must"
        dir_to_log = os.path.dirname(os.path.dirname(__file__))
        if hparams.exp_name is None:
            wandb_logger = WandbLogger(project="eend",entity='mlspeech', id=hparams.wandb_id, resume=resume)
        else:
            wandb_logger = WandbLogger(project="eend",entity='mlspeech', id=hparams.wandb_id, resume=resume, name=hparams.exp_name,
                                        settings=wandb.Settings(code_dir=dir_to_log))

        loggers_list.append(wandb_logger)

    elif hparams.tensorboard:
        tensorboard_logger = TensorBoardLogger(name=hparams.exp_name, save_dir="tensorboard_encoder")
        loggers_list.append(tensorboard_logger)

    else:
        logger = DummyLogger()
        loggers_list.append(logger)
    checkpoint = ModelCheckpoint(
        dirpath=hparams.model_save_dir,
        save_top_k=1,
        verbose=True,
        monitor=hparams.stop_measure,
        mode=hparams.stop_measure_type,
        every_n_epochs=1,
        every_n_train_steps=0
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks_list = [checkpoint, early_stop, lr_monitor]


    if hparams.initmodel and hparams.wandb_id:
        min_epochs = int(hparams.initmodel.split("/")[-1].split("=")[1].split("-")[0]) + 1
        ckpt_path = hparams.initmodel
        print("restore")
    else:
        min_epochs = 1
        ckpt_path = None
        print("new training")
    if hparams.gpu <= 0:
        accelerator = 'cpu'
        devices = 16
    else:
        accelerator = 'gpu'
        devices = hparams.gpu
    trainer = Trainer(
            logger=loggers_list,
            check_val_every_n_epoch=1,
            min_epochs=min_epochs,
            max_epochs=hparams.max_epochs,
            callbacks = callbacks_list,
            accelerator=accelerator,
            devices=devices,
            strategy="ddp",
            log_every_n_steps=30,
            # detect_anomaly=True
            replace_sampler_ddp = True, 
            accumulate_grad_batches=hparams.gradient_accumulation_steps,
            )

    if not hparams.test:

        trainer.fit(model, datamodule=my_data_moudule, ckpt_path=ckpt_path)
        if hparams.test_data_dir:
            torch.distributed.destroy_process_group()
            if trainer.is_global_zero:
                trainer = Trainer(
                    logger=loggers_list,
                    check_val_every_n_epoch=1,
                    min_epochs=min_epochs,
                    max_epochs=hparams.max_epochs,
                    callbacks = [checkpoint, early_stop, lr_monitor],
                    accelerator=accelerator,
                    devices=1,
                    strategy="ddp",
                    log_every_n_steps=30,
                    replace_sampler_ddp = True, 
                    accumulate_grad_batches=hparams.gradient_accumulation_steps,
                    )
                trainer.test(model, datamodule=my_data_moudule, ckpt_path="best")
    else:
        trainer.test(model, datamodule=my_data_moudule, )


