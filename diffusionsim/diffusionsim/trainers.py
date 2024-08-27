import time
import os
import sys
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import xarray as xr
from dataclasses import dataclass, asdict, field


def create_optimizer(model, config):
    match config.optimizer.lower():
        case "adam":
            try:
                my_betas = config.betas
            except AttributeError:
                my_betas = (0.9, 0.999) # default values
            optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=my_betas)

        case _: # defaults to SGD
            optim = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    return(optim)

class AbstractTrainer(ABC):
    def __init__(self, model, datasets, loss_fn, config, exp_id, rank=0):
        self.exp_id = exp_id
        self.training_config = config
        self.device, self.rank = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu', rank
        self.distributed = bool(config.distributed_training)
        self.model, self.datasets, self.loss_fn = model, datasets, loss_fn
        self.optimizer = create_optimizer(self.model, config)
        self.dataloaders = self._prepare_dataloaders()
        self._set_directories()
        self.phases = config.phases
        
        if(self.distributed):
            assert rank >= 0, "Need to pass in process rank"
            #self.distributed_backend = 'nccl' if self.device == 'cuda' else 'gloo'
            self.exp_id = "dist_" + self.exp_id
            self.model = DDP(self.model.to(self.device), device_ids=[rank])

    def _set_directories(self, directories='default'):
        base_dir = f"experiments/{self.exp_id}"
        if(directories == 'default'):
            self.dirs = dict(
                log_dir = base_dir,
                ckpt_dir = base_dir,
                output_dir = base_dir,
            )
        
    def _prepare_dataloaders(self):
        if(self.distributed):
            return(self._prepare_distributed_dataloaders())
        dataloaders = {}
        for phase, dataset in self.datasets.items():
            dataloaders[phase] = DataLoader( # add more args for data parallel or whatever down the line
                dataset, shuffle=self.training_config.shuffle_data[phase], batch_size=self.training_config.batch_size,
            )
        return(dataloaders)

    def _prepare_distributed_dataloaders(self):
        dataloaders = {}
        for phase, dataset in self.datasets.items():
            dataloaders[phase] = DataLoader( # add more args for data parallel or whatever down the line
                dataset, shuffle=False, batch_size=self.training_config.batch_size, rank = self.rank, 
                pin_memory = True, sampler = DistributedSampler(dataset)
            )
        return(dataloaders)
    
    def _save_checkpoint(self, epoch, cid=''):
        ckp = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        path = os.path.join(self.dirs['ckpt_dir'], f"{cid}-ckpt.pt")
        torch.save(ckp, path)
        print(f"Model saved at {path}")
    
    def _run_batch(self, batch, phase):
        data, label = batch
        output = self.model(data)
        loss = self.loss_fn(output, label)
        if(phase == 'train'):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return(output, loss)

    @abstractmethod
    def _run_epoch(self, epoch):
        pass

    @abstractmethod
    def _log_epoch_info(self, epoch_num, epoch_stats):
        pass
    
    def train(self, num_epochs, log=True, trial=''):
        self.setup_training(num_epochs)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("_" * 10)
            stats = self._run_epoch(epoch)
            self._log_epoch_info(epoch, stats)
        
        self.finish_training(log, trial)

    def setup_training(self, num_epochs):
        print("Getting ready to train model")
        for dirname in self.dirs.values():
            Path(dirname).mkdir(parents=True, exist_ok=True)
        self.losses = {}
        for phase in self.phases:
            self.losses[phase] = {}

    def finish_training(self, log, trial):
        print("Finished training")
        if(log):
            log_dict = {
                "training_config" : asdict(self.training_config), 
            }
            for phase in self.phases:
                log_dict[f'{phase}_loss'] = self.losses[phase]
            if(self.training_config.log_gradients):
                log_dict['gradients'] = self.gradients

            with open(os.path.join(self.dirs['log_dir'], f"trial{trial}_log.json"), 'w') as f:
                json.dump(log_dict, f)
            return(log_dict)
        
            


class VAETrainer(AbstractTrainer):
    def __init__(self, model, datasets, loss_fn, config, rank = 0, exp_id='VAE'):
        """
        Trainer Class for VAE Training
        """
        # abstract trainer: model, datasets, loss, config, exp, rank
        super().__init__(model, datasets, loss_fn, config, exp_id, rank)

    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        self.losses = {}
        for phase in self.phases:
            self.losses[phase] = {'mse':[], 'kl':[]}
        self.best_loss = 10
        if(self.training_config.log_gradients):
            self.gradients = dict(kl_gradients={}, mse_gradients = {})
            for name, param in self.model.named_parameters():
                self.gradients['kl_gradients'][name] = []
                self.gradients['mse_gradients'][name] = []

        self.batches_per_epoch = len(self.dataloaders['train'])

        self.eval_losses = dict(mse=[], kl=[])

    def _run_batch(self, batch, phase):
        x_hat = self.model(batch)
        kl_div = self.model.encoder.kl
        mse = self.loss_fn(batch, x_hat)# + self.training_config.beta * kl_div
        if(phase == 'train'):
            self.optimizer.zero_grad()
            if(self.training_config.log_gradients):
                # Compute and store gradients for MSE
                mse.backward(retain_graph=True)
                for name, p in self.model.named_parameters():
                    self.gradients['mse_gradients'][name].append(p.grad.mean().item())
                self.optimizer.zero_grad()  # Clear the gradients again
    
                # Compute and store gradients for KL divergence
                (self.training_config.beta*kl_div).backward(retain_graph=True)
                for name, p in self.model.encoder.named_parameters():
                    name = "encoder." + name
                    self.gradients['kl_gradients'][name].append(p.grad.mean().item())
                self.optimizer.zero_grad()  # Clear the gradients again
            
            # Combine the loss and backpropagate
            total_loss = mse + self.training_config.beta * kl_div
            total_loss.backward()
            self.optimizer.step()

        return(x_hat, mse, kl_div)

    def _run_train_epoch(self, epoch):
        mse_loss, kl_div = 0.0, 0.0
        self.model.train(True)
        bli = self.training_config.batch_logging_interval
        with torch.set_grad_enabled(True):
            if(self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(self.dataloaders['train']):
                x_hat, mse, kl = self._run_batch(batch, 'train')
                mse_loss += mse.item()
                kl_div += kl.item()
                if((step+1) % bli == 0):
                    mse_loss = mse_loss / bli
                    kl_div =  kl_div / bli
                    self._log_step('train', mse_loss, kl_div, step)
                    mse_loss, kl_div = 0.0, 0.0
            remainder_steps = self.batches_per_epoch % bli
            self._log_step('train', mse_loss/remainder_steps, kl_div/remainder_steps, self.batches_per_epoch)
        return(0)

    def _run_eval_epoch(self, epoch):
        phase = 'eval'
        mse_loss, kl_div = 0.0, 0.0
        self.model.train(False)
        with torch.set_grad_enabled(False):
            num_batches = len(self.dataloaders[phase])
            for step, batch in enumerate(self.dataloaders[phase]):
                x_hat, mse, kl = self._run_batch(batch, phase)
                mse_loss += mse.item()
                kl_div += kl.item()
        self.eval_losses['mse'].append( mse_loss / num_batches )
        self.eval_losses['kl'].append( kl_div / num_batches )
        return(0)
 
    def _log_epoch_info(self, epoch_num, epoch_stats):
        logs_per_epoch = self.batches_per_epoch // self.training_config.batch_logging_interval + 1
        for phase, loss_dict in self.losses.items():
            epoch_mse = sum(loss_dict['mse'][-logs_per_epoch:]) / logs_per_epoch
            epoch_kl = sum(loss_dict['kl'][-logs_per_epoch:]) / logs_per_epoch
            print(f"{phase} stats [mse : {epoch_mse}; kl : {epoch_kl}]")
            is_best_loss = (epoch_mse+epoch_kl)<self.best_loss
            if(is_best_loss and self.rank == 0 and phase == 'train'):
                print(f"saving new checkpoint at epoch {epoch_num}")
                self.best_loss = epoch_mse + epoch_kl
                self._save_checkpoint(epoch_num, cid='best') # TODO: I dont think this is running

    def _log_step(self, phase, mse, kl, step):
        bli = self.training_config.batch_logging_interval
        self.losses[phase]['mse'].append( mse )
        self.losses[phase]['kl'].append( kl )
        print(f"Batch {step}/{self.batches_per_epoch} [mse: {mse}  kl: {kl}]")
        if(self.training_config.log_gradients and False):
            for name, p in self.model.named_parameters():
                gnorm = torch.linalg.norm(p.grad)
                self.gradients[name].append(gnorm.detach().item())

    def _run_epoch(self, epoch):
        self._run_train_epoch(epoch)
        if(epoch % 2 == 0):
            self._run_eval_epoch(epoch)
        print(f"Eval Loss on epoch {epoch}: [mse : {self.eval_losses['mse'][-1]}, kl {self.eval_losses['kl'][-1]}]")



class DiffusionTrainer(AbstractTrainer):
    def __init__(self, model, datasets, loss_fn, tconfig, mconfig, exp_id, rank=0):
        super().__init__(model, datasets, loss_fn, tconfig, exp_id, rank)
        self.model_config = mconfig
    
    def _run_batch(self, batch, phase='train'):
        noisy_images, timesteps, noises = batch
        noise_pred = self.model(noisy_images, timesteps.flatten()).sample
        loss = self.loss_fn(noise_pred, noises)
        if(phase == 'train'):
            self.optimizer.zero_grad()
            loss.backward()
            if(self.training_config.clip_gradients):
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
    
        return(loss)

    def _run_epoch(self, epoch, phase='train'):
        self.model.train(phase=='train')
        total_loss = 0
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for step, clean_batch in enumerate(self.dataloaders['train']):
                batch = self.datasets['train'].noise_batch(clean_batch)
                loss = self._run_batch(batch, phase)
                total_loss += loss.item()
            
        return(total_loss / len(self.dataloaders['train']))
    
    def _log_epoch_info(self, epoch_num, loss):
        self.losses['train'].append(loss)
        if(epoch_num % self.training_config.epoch_logging_interval == 0):
            print(f"epoch {epoch_num}: [{loss}]")
        if(loss < self.best_loss and self.training_config.save_best_epoch):
            self.best_loss = loss
            self._save_checkpoint(epoch_num, cid='best')
        
        
    def train(self, num_epochs=0, log=True, trial=''):
        self.setup_training(num_epochs)
        print(f"Training for {self.training_config.num_epochs} epochs")
        for epoch in range(self.training_config.num_epochs):
            loss = self._run_epoch(epoch)
            self._log_epoch_info(epoch, loss)
        self.finish_training(log, trial)
    
    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        if(num_epochs): # passed in number of training epochs is non zero
            self.training_config.num_epochs = num_epochs
        for phase in self.phases:
            self.losses[phase] = []
        self.best_loss = 10

    def finish_training(self, log, trial):
        print("Finished training")
        if(log):
            log_dict = {
                "training_config" : asdict(self.training_config), 
                "model_config" : asdict(self.model_config)
            }
            for phase in self.phases:
                log_dict[f'{phase}_loss'] = self.losses[phase]
            #if(self.training_config.log_gradients):
            #    log_dict['gradients'] = self.gradients

            with open(os.path.join(self.dirs['log_dir'], f"trial_{trial}_log.json"), 'w') as f:
                json.dump(log_dict, f)
            return(log_dict)