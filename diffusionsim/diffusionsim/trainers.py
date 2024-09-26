import time
import os
import sys
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime
from pathlib import Path
import traceback
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import xarray as xr
from dataclasses import dataclass, asdict, field
from .mydatasets import log_event


def create_optimizer(model, tconfig):
    match tconfig.optimizer.lower():
        case "adam":
            try:
                my_betas = tconfig.betas
            except AttributeError:
                my_betas = (0.9, 0.999) # default values
            optim = torch.optim.Adam(model.parameters(), lr=tconfig.learning_rate, betas=my_betas)

        case _: # defaults to SGD
            optim = torch.optim.SGD(model.parameters(), lr=tconfig.learning_rate)
    return(optim)

class AbstractTrainer(ABC):
    def __init__(self, model, dataloader, loss_fn, optim, tconfig, rank=0, base_dir=''):
        self.exp_id = tconfig.exp_id
        self.training_config = tconfig
        self.device, self.rank = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu', rank
        self.distributed = bool(tconfig.distributed_training)
        self.model, self.dataloaders, self.loss_fn, self.optimizer = model, dict(train=dataloader), loss_fn, optim
        self._set_directories(base_dir=base_dir)
        self.current_run_id = ""
        if(self.distributed):
            assert rank >= 0, "Need to pass in process rank"
            #self.distributed_backend = 'nccl' if self.device == 'cuda' else 'gloo'
            self.exp_id = "dist_" + self.exp_id
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.to(self.device), device_ids=[rank])

    def _set_directories(self, directories='default', base_dir=''):
        if(not base_dir):
            base_dir = f"experiments/{self.exp_id}"
        if(directories == 'default'):
            self.dirs = dict(
                log_dir = base_dir,
                ckpt_dir = base_dir,
                output_dir = os.path.join(base_dir, "outputs"),
            )

    def _save_checkpoint(self, epoch, cid=''):
        ckp = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        path = os.path.join(self.dirs['ckpt_dir'], f"{cid}{self.current_run_id}-ckpt.pt")
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
    
    def train(self, num_epochs, log=True, run_id='trialx'):
        self.setup_training(num_epochs)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("_" * 10)
            stats = self._run_epoch(epoch)
            self._log_epoch_info(epoch, stats)
        
        self.finish_training(log, run_id, num_epochs)

    def setup_training(self, num_epochs, run_id):
        print(f"Getting ready to train model for {num_epochs} epochs")
        for dirname in self.dirs.values():
            Path(dirname).mkdir(parents=True, exist_ok=True)
        self.losses = dict(train=[], eval=[])
        self.log_file = os.path.join(self.dirs['log_dir'], f"{run_id}.json")
        if(num_epochs): # passed in number of training epochs is non zero
            self.training_config.num_epochs = num_epochs
        if(self.training_config.log_gradients):
            self.gradients = []
        self.current_run_id = run_id

    def finish_training(self, log, num_epochs):
        print(f"Finished training {num_epochs} epochs.")
        self.current_run_id = ""
        if(log):
            with open(self.log_file, "r") as configs:
                log_dict = json.load(configs)
                log_dict['training_config']['num_epochs'] = num_epochs
            
            for phase, loss in self.losses.items():
                log_dict[f'{phase}_loss'] = loss
                
            if(self.training_config.log_gradients):
                log_dict['gradients'] = self.gradients
            with open(self.log_file, 'w') as f:
                json.dump(log_dict, f)
            return(log_dict)
        
class VAETrainer(AbstractTrainer):
    def __init__(self, model, dataloader, loss_fn, optim, tconfig, rank=0, base_dir=''):
        """
        Trainer Class for VAE Training
        """
        super().__init__(model, dataloader, loss_fn, optim, tconfig, rank=0, base_dir='')

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

    def _run_train_epoch(self, epoch, phase='train'):
        mse_loss, kl_div = 0.0, 0.0
        self.model.train(True)
        bli = self.training_config.batch_logging_interval
        dataloader = self.dataloaders[phase]
        with torch.set_grad_enabled(True):
            if(self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(dataloader):
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
            for step, batch in enumerate(self.eval_dataloader):
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
    def __init__(self, model, scheduler, dataloader, loss_fn, optim, tconfig, rank=0, base_dir=''):
        super().__init__(model, dataloader, loss_fn, optim, tconfig, rank, base_dir)
        self.scheduler = scheduler
    
    def _run_batch(self, images, timesteps, noises, phase='train'):
        t0 = log_event("run-batch start")
        noise_pred = self.model(images, timesteps.flatten()).sample
        loss = self.loss_fn(noise_pred, noises)
        if(phase == 'train'):
            self.optimizer.zero_grad()
            loss.backward()
            if(self.training_config.clip_gradients):
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        log_event("run-batch end", duration=time.time() - t0)
        return(loss)

    def _run_epoch(self, epoch, phase='train'):
        self.model.train(phase=='train')
        self.epoch_losses = []
        total_loss = 0.0
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                self.dataloaders[phase].sampler.set_epoch(epoch)
            for step, images in enumerate(self.dataloaders[phase]):
                tt0 = log_event("training start", batch=step)
                # Given a batch from a dataloader on the dataset, return a noised sample
                noises = torch.randn(images.shape, device=self.device)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                size=(images.shape[0],), device=self.device, dtype=torch.int64)
                images = self.scheduler.add_noise(images.to(self.device), noises, timesteps)
                loss = self._run_batch(images, timesteps, noises, phase)
                total_loss += loss.item()
                total_loss = self.log_step(epoch, step, total_loss, phase)
                log_event("training end", batch=step, duration= time.time() - tt0)
                if(step % 50 == 0):
                    print(f"Currently at epoch {epoch}, step {step}")

        self.losses[phase] = self.losses[phase] + self.epoch_losses
        return(sum(self.epoch_losses) / len(self.epoch_losses))

    def log_step(self, epoch, step, total_loss, phase):
        bli = self.training_config.batch_logging_interval
        if((step+1) % bli == 0):
            self.epoch_losses.append(total_loss / bli)
            total_loss = 0.0

        if ((step+1) % self.training_config.batch_checkpoint_interval == 0):
            print(f"epoch {epoch}, step {step}: saving checkpoint")
            self._save_checkpoint(epoch, cid='')
        if(step + 1 == len(self.dataloaders[phase])):
            # last batch of epoch
            remainder_steps = len(self.dataloaders[phase]) % bli
            self.epoch_losses.append(total_loss / remainder_steps)
            total_loss = 0.0
        return(total_loss)
    
    def _log_epoch_info(self, epoch_num, loss):
        #self.losses['train'].append(loss)
        #if(epoch_num % self.training_config.epoch_logging_interval == 0):
        #    print(f"epoch {epoch_num}: [{loss}]")
        if(loss < self.best_loss and self.training_config.save_best_epoch):
            self.best_loss = loss
            self._save_checkpoint(epoch_num, cid='best_')
    
    #def _save_checkpoint(self, epoch, cid):
        # extend such that training can be resumed from checkpoint alone. 
        # possibly create and delete as part of finish training? 
    #    super()._save_checkpoint(epoch,  cid=cid)
     
    def train(self, num_epochs=0, log=True, run_id=''):
        self.setup_training(num_epochs, run_id)
        print(f"Training for {self.training_config.num_epochs} epochs")
        try: 
            for epoch in range(self.training_config.num_epochs):
                e0 = log_event("epoch start", epoch=epoch)
                loss = self._run_epoch(epoch)
                self._log_epoch_info(epoch, loss)
                log_event("epoch end", epoch=epoch, duration=time.time() - e0, epoch_loss=loss)
        except Exception as e:
            print(f"Some error occurred during training: {e}")
            traceback.print_exc(file=sys.stdout)
        self.finish_training(log, num_epochs)
    
    def setup_training(self, num_epochs, run_id):
        super().setup_training(num_epochs, run_id)
        self.best_loss = 100
        self.model = self.model.to(self.device)
        self.event_file = 0 #open(os.path.join(self.dirs['output_dir'], "event_log.txt"), "a")


    def finish_training(self, log, num_epochs):
        # to do: add github hash as well so can reproduce code base at time of training run
        super().finish_training(log, num_epochs)
        #if(self.event_file):
        #    self.event_file.close()