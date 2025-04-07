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
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank, **kwargs):
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        self.rank = rank
        self.phases = tconfigs[0].phases
        assert isinstance(mconfigs, list) and isinstance(tconfigs, list) and len(mconfigs) == len(tconfigs), "mconfigs and tconfigs must be lists of same length"
        self.dconfig = dconfig
        dls, indices = tru.load_dataloaders(dconfig)
        self.dataloaders = {}
        for i, phase in enumerate(self.phases):
            self.dataloaders[phase] = dls[i]
        
        self.number_of_models = len(mconfigs)
        self.run_ids = [tconfig.run_id for tconfig in tconfigs]
        self.mconfigs = dict(zip(self.run_ids, mconfigs))
        self.training_configs = dict(zip(self.run_ids, tconfigs))
        self.distributed = bool(tconfigs[0].distributed_training)
        self.models = dict(zip(self.run_ids, [tru.load_model(mconfig, device=self.device, distributed=self.distributed) for mconfig in mconfigs]))
        optimizers = [tru.create_optimizer(model, tconfig) for model, tconfig in zip(self.models.values(), tconfigs)]
        self.optimizers = dict(zip(self.run_ids, optimizers))
        self.loss_fn = loss_fn
        if("dirs" in kwargs):
            self.dirs = kwargs['dirs']
        else:
            self.dirs = dict(
                log_dir = base_dir,
                ckpt_dir = os.path.join(base_dir, "checkpoints"),
                output_dir = os.path.join(base_dir, "outputs"),
            )
        
        self.bli = tconfigs[0].batch_logging_interval
        self.ckpt_interval = tconfigs[0].batch_checkpoint_interval


    def _save_checkpoint(self, run_id, cid='', **kwargs):
        model = self.models[run_id]
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
            
            batch_results[self.run_ids[idx]] = {
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
    
    def train(self, num_epochs, log=True):
        self.setup_training(num_epochs)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("_" * 10)
            stats = self._run_epoch(epoch)
            self._log_epoch_info(stats, epoch, phase="train")
        
        self.finish_training(num_epochs)

    def setup_training(self, num_epochs):
        print(f"Getting ready to train {self.number_of_models} model(s) for {num_epochs} epochs")
        for dirname in self.dirs.values():
            Path(dirname).mkdir(parents=True, exist_ok=True)
        self.batches_per_epoch = dict(zip(self.phases, [len(self.dataloaders[phase]) for phase in self.phases]))

        self.losses, self.gradients, self.best_losses = {}, {}, {}
        for run_id in self.run_ids:
            tconfig, mconfig = self.training_configs[run_id], self.mconfigs[run_id]
            tconfig.num_epochs = num_epochs
            self.losses[run_id] = {phase: [] for phase in self.phases}
            if(tconfig.log_gradients):
                self.gradients[run_id] = []
            self.best_losses[run_id] = float('inf')

            log_file = os.path.join(self.dirs['log_dir'], f"{run_id}.json")
            with open(log_file, "w") as f:
                json.dump(dict(
                    training_config=dataclasses.asdict(tconfig),
                    model_config=dataclasses.asdict(mconfig),
                    data_config=dataclasses.asdict(self.dconfig),
                ), f)

    def finish_training(self, num_epochs, **kwargs):
        print(f"Finished training {self.number_of_models} model(s) for {num_epochs} epochs.")

        for i, run_id in enumerate(self.run_ids):
            log_file = os.path.join(self.dirs['log_dir'], f"{run_id}.json")
            with open(log_file, "r+") as f:
                log_dict = json.load(f)
                
                # Add losses for this model
                log_dict['best_loss'] = self.best_losses[run_id]
                log_dict['losses'] = self.losses[run_id]
                for phase, losses in self.losses[run_id].items():
                    log_dict[f'{phase}_loss'] = losses
                
                # Add any additional kwargs
                for key, value in kwargs.items():
                    if(isinstance(value, dict)):
                        for k, v in value.items():
                            if(k not in self.run_ids):
                                log_dict[k] = v
                            elif(k == run_id):
                                # this assumes that the arg was argname={run_id: value for run_id}
                                log_dict[key] = v
                    else:
                        log_dict[key] = value
                if self.training_configs[run_id].log_gradients:
                    log_dict['gradients'] = self.gradients[run_id]
                
                # Reset file pointer to beginning and truncate
                f.seek(0)
                json.dump(log_dict, f)
                f.truncate()
                    
        return

class ClimsimTrainer(AbstractTrainer):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank=0, **kwargs):
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank)
        self.loss_weights = {}
        self.tracked_losses = ["mse", "total"]
        for run_id in self.run_ids:
            loss_weights = self.training_configs[run_id].loss_weights
            if(loss_weights['distribution'] > 0):
                self.tracked_losses.append("distribution")
            if(loss_weights['diffusion'] > 0):
                self.tracked_losses.append("diffusion")
                assert 'unet' in kwargs and 'scheduler' in kwargs, "Need to pass in diffusion model"
                self.unet = kwargs['unet']
                self.scheduler = kwargs['scheduler']
            

            self.loss_weights[run_id] = loss_weights

    
    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        for run_id in self.run_ids:
            for loss_type in self.tracked_losses:
                self.losses[run_id][loss_type] = []

    def _run_epoch(self, epoch, phase='train'):
        for run_id in self.run_ids:
            self.models[run_id].train(phase=='train')

        """
        Tracking 3 kinds of losses, which can be confusing: 
        batch_losses: losses for a single batch. Because there might be many batches in an epoch, do not save every single batch loss
        current_losses: instead, divide epoch into batch_logging_interval sized sections, and save the mean of the losses for each section
        epoch_losses: the number of loss items saved for a single epoch is (batches_per_epoch / batch_logging_interval) 
        """
        tru.log_event(f"epoch-{epoch} start")
        epoch_losses = defaultdict(dict)
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                self.dataloaders[phase].sampler.set_epoch(epoch)
            steps_per_epoch = len(self.dataloaders[phase])
            for step, (X, Y) in enumerate(self.dataloaders[phase]):
                if(step % 10 == 0):
                    print(f"Currently at epoch {epoch}, step {step}/{steps_per_epoch}")                
                batch_losses = self._run_batch(X.to(self.device), Y.to(self.device), phase)
                epoch_losses = self.log_step(epoch_losses, batch_losses, step, phase)
                
                if ((step+1) % self.ckpt_interval == 0 and phase == 'train'):
                    print(f"epoch {epoch}, step {step}: saving checkpoint")
                    for run_id in self.run_ids:
                        self._save_checkpoint(run_id)
                
        tru.log_event(f"epoch-{epoch} end")
        return(epoch_losses)

    def log_step(self, epoch_losses, batch_losses, step, phase):
        for run_id in self.run_ids:
            for loss_type in self.tracked_losses:
                if(step == 0):
                    epoch_losses[run_id][loss_type] = []
                if(step % self.bli == 0):
                    epoch_losses[run_id][loss_type].append(batch_losses[run_id][loss_type])


        return(epoch_losses)

    def _log_epoch_info(self, epoch_stats, epoch_num, phase='train'):
        logs_per_epoch = self.batches_per_epoch[phase] // self.bli
        print("\n\n")
        for run_id in self.run_ids:
            for loss_type in self.tracked_losses:
                avg_loss_value = sum(epoch_stats[run_id][loss_type]) / len(epoch_stats[run_id][loss_type])
                self.losses[run_id][loss_type] = self.losses[run_id][loss_type] + epoch_stats[run_id][loss_type]
                print(f"avg {loss_type} loss for epoch {epoch_num}, run_id {run_id}: {avg_loss_value}")
                if(loss_type == 'total' and avg_loss_value < self.best_losses[run_id] and phase == 'train'):
                    print(f"saving new checkpoint at epoch {epoch_num}")
                    self.best_losses[run_id] = avg_loss_value
                    self._save_checkpoint(run_id)
                    

    def _run_batch(self, x, y, phase):
        t0 = tru.log_event("run-batch start")
        batch_losses = {}
        for run_id in self.run_ids:
            batch_losses[run_id] = {}
            model, optimizer, loss_weights = self.models[run_id], self.optimizers[run_id], self.loss_weights[run_id]
            y_hat = model(x)
            mse_loss = self.loss_fn(y_hat, y)
            total_loss = loss_weights['mse'] * mse_loss
            batch_losses[run_id]['mse'] = mse_loss.item()
            if(loss_weights['distribution'] > 0):
                dist_loss = self.distribution_loss(y, y_hat)
                total_loss += loss_weights['distribution'] * dist_loss
                batch_losses[run_id]['distribution'] = dist_loss.item()
            if(loss_weights['diffusion'] > 0):
                diff_loss = self.diffusion_loss(y, y_hat)
                total_loss += loss_weights['diffusion'] * diff_loss
                batch_losses[run_id]['diffusion'] = diff_loss.item()

            batch_losses[run_id]['total'] = total_loss.item()
        
            if(phase == 'train'):
                optimizer.zero_grad()
                total_loss.backward()
                if(self.training_configs[run_id].clip_gradients):
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
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

class VAETrainer(AbstractTrainer):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank=0):
        """
        Trainer Class for VAE Training
        """
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank)

    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        self.losses = {}
        for phase in self.phases:
            self.losses[phase] = {'mse':[], 'kl':[]}
        self.best_loss = 10
        if(self.training_configs[self.run_ids[0]].log_gradients):
            self.gradients = dict(kl_gradients={}, mse_gradients = {})
            for name, param in self.model.named_parameters():
                self.gradients['kl_gradients'][name] = []
                self.gradients['mse_gradients'][name] = []

        self.batches_per_epoch = len(self.dataloaders['train'])

        self.eval_losses = dict(mse=[], kl=[])

    def _run_batch(self, batch, phase):
        for run_id in self.run_ids:
            model, optimizer, tconfig = self.models[run_id], self.optimizers[run_id], self.training_configs[run_id]
            x_hat = model(batch)
            kl_div = model.encoder.kl
            mse = self.loss_fn(batch, x_hat)# + self.training_configs[run_id].beta * kl_div
            if(phase == 'train'):
                optimizer.zero_grad()
                if(tconfig.log_gradients):
                    # Compute and store gradients for MSE
                    mse.backward(retain_graph=True)
                    for name, p in model.named_parameters():
                        self.gradients['mse_gradients'][name].append(p.grad.mean().item())
                    optimizer.zero_grad()  # Clear the gradients again
    
                    # Compute and store gradients for KL divergence
                    (tconfig.beta*kl_div).backward(retain_graph=True)
                    if(tconfig.log_gradients):
                        for name, p in model.encoder.named_parameters():
                            name = "encoder." + name
                        self.gradients['kl_gradients'][name].append(p.grad.mean().item())
                    optimizer.zero_grad()  # Clear the gradients again

            # Combine the loss and backpropagate
                total_loss = mse + tconfig.beta * kl_div
                total_loss.backward()
                optimizer.step()

        return(x_hat, mse, kl_div)

    def _run_train_epoch(self, epoch, phase='train'):
        mse_loss, kl_div = 0.0, 0.0
        self.model.train(True)
        dataloader = self.dataloaders[phase]
        with torch.set_grad_enabled(True):
            if(self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(dataloader):
                x_hat, mse, kl = self._run_batch(batch, 'train')
                mse_loss += mse.item()
                kl_div += kl.item()
                if((step+1) % self.bli == 0):
                    mse_loss = mse_loss / self.bli
                    kl_div =  kl_div / self.bli
                    self._log_step('train', mse_loss, kl_div, step)
                    mse_loss, kl_div = 0.0, 0.0
            remainder_steps = self.batches_per_epoch % self.bli
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
        logs_per_epoch = self.batches_per_epoch // self.bli + 1
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
        self.losses[phase]['mse'].append( mse )
        self.losses[phase]['kl'].append( kl )
        print(f"Batch {step}/{self.batches_per_epoch} [mse: {mse}  kl: {kl}]")
        if(self.training_configs[self.run_ids[0]].log_gradients and False):
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
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank=0):
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank)
        self.scheduler = tru.load_scheduler(mconfigs[0])
    
    def _run_batch(self, images, phase='train'):
        t0 = tru.log_event("run-batch start")
        batch_losses = {}
        for run_id in self.run_ids:
            model, optimizer, tconfig = self.models[run_id], self.optimizers[run_id], self.training_configs[run_id]
            noises = torch.randn(images.shape, device=self.device)
            timesteps = torch.randint(0, tconfig.max_T_sample, 
                                      size=(images.shape[0],), device=self.device, dtype=torch.int64)
            self.noise_timesteps[run_id] += torch.bincount(timesteps.cpu(), minlength=tconfig.max_T_sample)
            images = self.scheduler.add_noise(images, noises, timesteps)
            noise_pred = model(images, timesteps.flatten()).sample
            loss = self.loss_fn(noise_pred, noises)
            if(phase == 'train'):
                optimizer.zero_grad()
                loss.backward()
                if(tconfig.clip_gradients):
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_losses[run_id] = loss.item()
        tru.log_event("run-batch end", duration=time.time() - t0, **batch_losses)
        return(batch_losses)

    def _run_epoch(self, epoch, phase='train'):
        for run_id in self.run_ids:
            self.models[run_id].train(phase=='train')
        epoch_losses = defaultdict(list)
        #total_loss = 0.0
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                self.dataloaders[phase].sampler.set_epoch(epoch)
            for step, images in enumerate(self.dataloaders[phase]):
                tt0 = tru.log_event(f"batch-{step} start", batch=step)
                # Given a batch from a dataloader on the dataset, return a noised sample
                batch_losses = self._run_batch(images.to(self.device), phase)
                #total_loss += loss.item()
                epoch_losses = self.log_step(epoch, step, batch_losses, phase, epoch_losses)
                tru.log_event(f"batch-{step} end", batch=step, duration= time.time() - tt0)
                if(step % 50 == 0):
                    print(f"Currently at epoch {epoch}, step {step}")
        return(epoch_losses)

    def log_step(self, epoch, step, batch_losses, phase, epoch_losses):
        step = step + 1
        for run_id in self.run_ids:
            if(step % self.bli == 0 or step == len(self.dataloaders[phase])):
                epoch_losses[run_id].append(batch_losses[run_id])

        if ((step+1) % self.ckpt_interval == 0):
            print(f"epoch {epoch}, step {step}: saving checkpoint")
            for run_id in self.run_ids:
                self._save_checkpoint(run_id)
        return(epoch_losses)
    
    def _log_epoch_info(self, epoch_num, epoch_losses, phase='train'):
        for run_id in self.run_ids:
            self.losses[run_id][phase] = self.losses[run_id][phase] + epoch_losses[run_id]
            avg_loss = sum(epoch_losses[run_id]) / len(epoch_losses[run_id])
            if(avg_loss < self.best_losses[run_id] and self.training_configs[run_id].save_best_epoch):
                self.best_losses[run_id] = avg_loss
                self._save_checkpoint(run_id, cid='best')
        
    #def _save_checkpoint(self, epoch, cid):
        # extend such that training can be resumed from checkpoint alone. 
        # possibly create and delete as part of finish training? 
    #    super()._save_checkpoint(epoch,  cid=cid)
     
    def train(self, num_epochs=0, log=True):
        self.setup_training(num_epochs)
        print(f"Training for {num_epochs} epochs")
        try: 
            for epoch in range(num_epochs): # set in setup_training
                e0 = tru.log_event("epoch start", epoch=epoch)
                loss = self._run_epoch(epoch)
                self._log_epoch_info(epoch, loss, phase='train')
                tru.log_event("epoch end", epoch=epoch, duration=time.time() - e0, epoch_loss=loss)
        except Exception as e:
            print(f"Some error occurred during training: {e}")
            traceback.print_exc(file=sys.stdout)
        self.finish_training(num_epochs)
    
    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        self.event_file = 0 #open(os.path.join(self.dirs['output_dir'], "event_log.txt"), "a")
        self.noise_timesteps = {}
        for run_id in self.run_ids:
            self.noise_timesteps[run_id] = torch.zeros(self.training_configs[run_id].max_T_sample, dtype=torch.long)
        

    def finish_training(self, num_epochs, **kwargs):
        # to do: add github hash as well so can reproduce code base at time of training run
        noise_timesteps = {}
        for run_id in self.run_ids:
            noise_timesteps[run_id] = self.noise_timesteps[run_id].numpy().tolist()
        super().finish_training(num_epochs, sampled_timesteps=noise_timesteps)
        #if(self.event_file):
        #    self.event_file.close()