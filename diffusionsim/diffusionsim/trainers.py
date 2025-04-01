import time
import os
import sys
import json
from abc import ABC, abstractmethod
from pathlib import Path
import traceback
import torch
import diffusionsim.training_utils as tru
from collections import defaultdict
import dataclasses


class AbstractTrainer(ABC):
    def __init__(self, dataloaders, mconfigs, tconfigs, loss_fn, base_dir, rank, **kwargs):
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        self.rank = rank
        assert isinstance(mconfigs, list) and isinstance(tconfigs, list) and len(mconfigs) == len(tconfigs), "mconfigs and tconfigs must be lists of same length"
        assert isinstance(dataloaders, dict), "dataloaders must be a dict <phase : dl>"
        
        self.number_of_models = len(mconfigs)
        self.training_configs = tconfigs
        self.distributed = bool(tconfigs[0].distributed_training)
        self.models = [tru.load_model(mconfig, device=self.device, distributed=self.distributed) for mconfig in mconfigs]
        self.optimizers = [tru.create_optimizer(m, tconfig) for m, tconfig in zip(self.models, self.tconfigs)]
        self.run_ids = [tconfig.run_id for tconfig in tconfigs]
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        if("dirs" in kwargs):
            self.dirs = kwargs['dirs']
        else:
            self.dirs = dict(
                log_dir = base_dir,
                ckpt_dir = os.path.join(base_dir, "checkpoints"),
                output_dir = os.path.join(base_dir, "outputs"),
            )
        

    def _save_checkpoint(self, run_id, cid='', **kwargs):
        model = self.models[self.run_ids.index(run_id)]
        ckp = model.module.state_dict() if self.distributed else model.state_dict()
        path = os.path.join(self.dirs['ckpt_dir'], f"{cid}{run_id}-ckpt.pt")
        torch.save(ckp, path)
        print(f"Model saved at {path}")
    
    def _run_batch(self, batch, phase):
        batch_results = {}
        for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            data, label = batch
            output = model(data)
            loss = self.loss_fn(output, label)
            
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            batch_results[self.model_ids[idx]] = {
                'output': output,
                'loss': loss.item()
            }
        
        return batch_results

    @abstractmethod
    def _run_epoch(self, epoch, phase):
        pass

    @abstractmethod
    def _log_epoch_info(self, epoch_num, epoch_stats, phase):
        pass
    
    def train(self, num_epochs, log=True, run_id='trialx'):
        self.setup_training(num_epochs, run_id)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("_" * 10)
            stats = self._run_epoch(epoch)
            self._log_epoch_info(epoch, stats, phase="train")
        
        self.finish_training(log, num_epochs)

    def setup_training(self, num_epochs, ):
        print(f"Getting ready to train {self.number_of_models} model(s) for {num_epochs} epochs")
        for dirname in self.dirs.values():
            Path(dirname).mkdir(parents=True, exist_ok=True)

        self.batches_per_epoch = len(self.dataloaders['train'])

        self.losses = {}
        self.gradients = {}
        for i, run_id in enumerate(self.run_ids):
            tconfig = self.training_configs[i]
            tconfig.num_epochs = num_epochs
            self.losses[run_id] = {phase: [] for phase in tconfig.phases}
            log_file = os.path.join(self.dirs['log_dir'], f"{run_id}.json")
            if(tconfig.log_gradients):
                self.gradients[run_id] = []
            self.best_losses[run_id] = float('inf')

        with open(os.path.join(self.dirs['log_dir'], f'{run_id}.json'), "w") as f:
            json.dump(dict(
                training_config=asdict(self.training_configs[0]),
                model_config=asdict(self.models[0]),
                data_config=asdict(self.dataloaders['train']),
            ), f)

    def finish_training(self, log, num_epochs, **kwargs):
        print(f"Finished training {self.number_of_models} model(s) for {num_epochs} epochs.")
        if not log:
            return
            
        log_dicts = {}
        for model_id in self.model_ids:
            with open(self.log_files[model_id], "r") as configs:
                log_dict = json.load(configs)
                log_dict['training_config']['num_epochs'] = num_epochs
                
                # Add losses for this model
                for phase, losses in self.losses[model_id].items():
                    log_dict[f'{phase}_loss'] = losses
                
                # Add any additional kwargs
                for key, value in kwargs.items():
                    if isinstance(value, dict) and model_id in value:
                        log_dict[key] = value[model_id]
                    
                if self.training_configs[0].log_gradients:
                    log_dict['gradients'] = self.gradients[model_id]
                    
                with open(self.log_files[model_id], 'w') as f:
                    json.dump(log_dict, f)
                    
                log_dicts[model_id] = log_dict
                
        return log_dicts




class ClimsimTrainer(AbstractTrainer):
    def __init__(self, dataloaders, loss_fn, optim, tconfig, base_dir, use_dist_loss=False, use_diff_loss=False, rank=0, **kwargs):
        super().__init__(dataloaders, mconfigs, tconfigs, loss_fn, base_dir, rank)
        self.use_dist_loss = use_dist_loss
        self.use_diff_loss = use_diff_loss
        self.lambda_0 = tconfig.loss_weights['mse']
        self.tracked_losses = ["mse", "total"]

        if(use_dist_loss):
            self.lambda_1 = tconfig.loss_weights['distribution']
            self.tracked_losses.append("distribution")

        if(use_diff_loss):
            assert 'unet' in kwargs, "Need to pass in diffusion model"
            assert 'scheduler' in kwargs, "Need to pass in scheduler"
            self.unet = kwargs['unet']
            self.scheduler = kwargs['scheduler']
            self.tracked_losses.append("diffusion")
            self.lambda_2 = tconfig.loss_weights['diffusion']
    
    def setup_training(self, num_epochs, run_id):
        super().setup_training(num_epochs, run_id)
        self.losses = dict()
        for loss_type in self.tracked_losses:
            self.losses[loss_type] = []

    def _run_epoch(self, epoch, phase='train'):
        self.model.train(phase=='train')

        """
        Tracking 3 kinds of losses, which can be confusing: 
        batch_losses: losses for a single batch. Because there might be many batches in an epoch, do not save every single batch loss
        current_losses: instead, divide epoch into batch_logging_interval sized sections, and save the mean of the losses for each section
        epoch_losses: the number of loss items saved for a single epoch is (batches_per_epoch / batch_logging_interval) 
        """
        tru.log_event(f"epoch-{epoch} start")
        epoch_losses = {}
        current_losses = {}
        for loss in self.tracked_losses:
            epoch_losses[loss] = []
            current_losses[loss] = 0.0
        
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                self.dataloaders[phase].sampler.set_epoch(epoch)
            steps_per_epoch = len(self.dataloaders[phase])
            for step, (X, Y) in enumerate(self.dataloaders[phase]):
                if(step % 10 == 0):
                    print(f"Currently at epoch {epoch}, step {step}/{steps_per_epoch}")                
                batch_losses = self._run_batch(X.to(self.device), Y.to(self.device), phase)
                epoch_losses, current_losses = self.log_step(epoch_losses, current_losses, batch_losses, epoch, step, phase)
        tru.log_event(f"epoch-{epoch} end")
        return(epoch_losses)

    def log_step(self, epoch_losses, current_losses, batch_losses, epoch, step, phase):
        bli = self.training_config.batch_logging_interval
        for loss_type in self.tracked_losses:
            current_losses[loss_type] += batch_losses[loss_type]
            if((step+1) % bli == 0):
                # entering new batch logging section, save avg and reset total counter
                epoch_losses[loss_type].append(current_losses[loss_type] / bli)
                current_losses[loss_type] = 0.0
            if(step + 1 == len(self.dataloaders[phase])):
                # last batch of epoch
                remainder_steps = len(self.dataloaders[phase]) % bli
                epoch_losses[loss_type].append(current_losses[loss_type] / remainder_steps)
                current_losses[loss_type] = 0.0

        if ((step+1) % self.training_config.batch_checkpoint_interval == 0 and phase == 'train'):
            print(f"epoch {epoch}, step {step}: saving checkpoint")
            self._save_checkpoint(epoch, cid=self.run)
        return(epoch_losses, current_losses)

    def _log_epoch_info(self, epoch_num, epoch_stats, phase='train'):
        logs_per_epoch = self.batches_per_epoch // self.training_config.batch_logging_interval + 1
        for loss_type in self.tracked_losses:
            avg_loss_value = sum(epoch_stats[loss_type]) / len(epoch_stats[loss_type])
            self.losses[loss_type] = self.losses[loss_type] + epoch_stats[loss_type]
            print(f"avg {loss_type} loss for epoch {epoch_num}: {avg_loss_value}")
            if(loss_type == 'total' and avg_loss_value < self.best_losses[self.model_ids[0]] and self.rank == 0 and phase == 'train'):
                print(f"saving new checkpoint at epoch {epoch_num}")
                self.best_losses[self.model_ids[0]] = avg_loss_value
                self._save_checkpoint(epoch_num, cid='best') # TODO: I dont think this is running

    def _run_batch(self, x, y, phase):
        t0 = tru.log_event("run-batch start")
        y_hat = self.model(x)
        mse_loss = self.loss_fn(y_hat, y)
        batch_losses = dict(mse=mse_loss.item())
        total_loss = self.lambda_0 * mse_loss
        if(self.use_dist_loss):
            dist_loss = self.distribution_loss(y, y_hat)
            total_loss += self.lambda_1 * dist_loss
            batch_losses['distribution'] = dist_loss.item()
        
        if(self.use_diff_loss):
            diff_loss = self.diffusion_loss(y, y_hat, T)
            total_loss += self.lambda_2 * diff_loss
            batch_losses['diffusion'] = diff_loss.item()

        batch_losses['total'] = total_loss.item()
        
        if(phase == 'train'):
            self.optimizer.zero_grad()
            total_loss.backward()
            if(self.training_config.clip_gradients):
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        tru.log_event("run-batch end", duration=time.time() - t0, loss=batch_losses['total'])
        return(batch_losses)
    
    def distribution_loss(self, y, yhat):
        raise NotImplementedError("Distribution loss not implemented")
    
    def diffusion_loss(self, y, yhat, T: int):
        #raise NotImplementedError("Diffusion loss not implemented")
        # TODO: y and yhat are of size (B, 128) and somehow need to convert into images for diffusion loss
        y, yhat = construct_image(y), construct_image(yhat)
        def encode(sample):
            eps = torch.randn(sample.shape, device=device) # BS x C x H x W
            xt = self.scheduler.add_noise(sample, eps, torch.LongTensor([T])) # noisy image
            return(xt)
        def decode(xt):
            for t in range(T, 0, -1):
                with torch.no_grad():
                    eps_theta = self.unet(xt, t).sample
                xt = self.scheduler.step(eps_theta, t, xt).prev_sample
            return(xt)
        yhat = decode(encode(yhat))
        return(self.image_loss_fn(y, yhat))


        if(phase == 'train'):
            self.optimizer.zero_grad()
            loss.backward()
            if(self.training_config.clip_gradients):
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        return(loss)  

class VAETrainer(AbstractTrainer):
    def __init__(self, dataloaders, mconfigs, tconfigs, loss_fn, base_dir, rank=0):
        """
        Trainer Class for VAE Training
        """
        super().__init__(dataloaders, mconfigs, tconfigs, loss_fn, base_dir, rank)

    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        self.losses = {}
        for phase in self.training_config.phases:
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
 
    def _log_epoch_info(self, epoch_num, epoch_stats, phase='train'):
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

    def _run_epoch(self, epoch, phase='train'):
        if(phase == 'train'):
            self._run_train_epoch(epoch)
        else:
            self._run_eval_epoch(epoch)
        print(f"Eval Loss on epoch {epoch}: [mse : {self.eval_losses['mse'][-1]}, kl {self.eval_losses['kl'][-1]}]")

class DiffusionTrainer(AbstractTrainer):
    def __init__(self, dataloaders, mconfigs, tconfigs, loss_fn, base_dir, rank=0):
        super().__init__(dataloaders, mconfigs, tconfigs, loss_fn, base_dir, rank)
        self.scheduler = scheduler

    
    def _run_batch(self, images, timesteps, noises, phase='train'):
        t0 = tru.log_event("run-batch start")
        noise_pred = self.model(images, timesteps.flatten()).sample
        loss = self.loss_fn(noise_pred, noises)
        if(phase == 'train'):
            self.optimizer.zero_grad()
            loss.backward()
            if(self.training_config.clip_gradients):
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        tru.log_event("run-batch end", duration=time.time() - t0, loss=loss.item())
        return(loss)

    def _run_epoch(self, epoch, phase='train'):
        self.model.train(phase=='train')
        self.epoch_losses = []
        total_loss = 0.0
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                self.dataloaders[phase].sampler.set_epoch(epoch)
            for step, images in enumerate(self.dataloaders[phase]):
                tt0 = tru.log_event(f"batch-{step} start", batch=step)
                # Given a batch from a dataloader on the dataset, return a noised sample
                noises = torch.randn(images.shape, device=self.device)
                timesteps = torch.randint(0, self.training_config.max_T_sample, 
                                size=(images.shape[0],), device=self.device, dtype=torch.int64)
                # Vectorized timestep counting
                self.noise_timesteps += torch.bincount(timesteps, minlength=self.training_config.max_T_sample)
                
                images = self.scheduler.add_noise(images.to(self.device), noises, timesteps)
                loss = self._run_batch(images, timesteps, noises, phase)
                total_loss += loss.item()
                total_loss = self.log_step(epoch, step, total_loss, phase)
                tru.log_event(f"batch-{step} end", batch=step, duration= time.time() - tt0)
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
    
    def _log_epoch_info(self, epoch_num, loss, phase='train'):
        #self.losses['train'].append(loss)
        #if(epoch_num % self.training_config.epoch_logging_interval == 0):
        #    print(f"epoch {epoch_num}: [{loss}]")
        if(loss < self.best_losses[self.model_ids[0]] and self.training_config.save_best_epoch):
            self.best_losses[self.model_ids[0]] = loss
            self._save_checkpoint(epoch_num, cid='best_')
    
    #def _save_checkpoint(self, epoch, cid):
        # extend such that training can be resumed from checkpoint alone. 
        # possibly create and delete as part of finish training? 
    #    super()._save_checkpoint(epoch,  cid=cid)
     
    def train(self, num_epochs=0, log=True, run_id=''):
        self.setup_training(num_epochs, run_id)
        print(f"Training for {self.training_config.num_epochs} epochs")
        try: 
            for epoch in range(self.training_config.num_epochs): # set in setup_training
                e0 = tru.log_event("epoch start", epoch=epoch)
                loss = self._run_epoch(epoch)
                self._log_epoch_info(epoch, loss, phase='train')
                tru.log_event("epoch end", epoch=epoch, duration=time.time() - e0, epoch_loss=loss)
        except Exception as e:
            print(f"Some error occurred during training: {e}")
            traceback.print_exc(file=sys.stdout)
        self.finish_training(log, num_epochs)
    
    def setup_training(self, num_epochs, run_id):
        super().setup_training(num_epochs, run_id)
        self.best_losses = {model_id: 100 for model_id in self.model_ids}
        self.event_file = 0 #open(os.path.join(self.dirs['output_dir'], "event_log.txt"), "a")
        self.noise_timesteps = torch.zeros(self.training_config.max_T_sample, device=self.device, dtype=torch.long)

    def finish_training(self, log, num_epochs):
        # to do: add github hash as well so can reproduce code base at time of training run
        noise_timesteps = self.noise_timesteps.cpu().numpy().tolist()
        super().finish_training(log, num_epochs, sampled_timesteps=noise_timesteps)
        #if(self.event_file):
        #    self.event_file.close()