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
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import random


class AbstractTrainer(ABC):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank, **kwargs):
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        self.rank = rank
        self.phases = tconfigs[0].phases
        assert isinstance(mconfigs, list) and isinstance(tconfigs, list) and len(mconfigs) == len(tconfigs), "mconfigs and tconfigs must be lists of same length"
        self.dconfig = dconfig
        dls, self.indices = tru.load_dataloaders(dconfig, log=True)
        self.dataloaders = {}
        for i, phase in enumerate(self.phases):
            self.dataloaders[phase] = dls[i]
        
        self.number_of_models = len(mconfigs)
        self.run_ids = [tconfig.run_id for tconfig in tconfigs]
        self.mconfigs = dict(zip(self.run_ids, mconfigs))
        self.training_configs = dict(zip(self.run_ids, tconfigs))
        self.distributed = bool(tconfigs[0].distributed_training)
        self.models = dict(zip(self.run_ids, [tru.load_model(mconfig, model_type=mconfig.model_type, device=self.device, distributed=self.distributed) for mconfig in mconfigs]))
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
        ckp = {k: v.detach().cpu() for k, v in ckp.items()}
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
            self.losses[run_id] = {phase: {} for phase in self.phases}
            if(tconfig.log_gradients):
                self.log_gradients = True
                self.gradients[run_id] = {}
            self.best_losses[run_id] = {}

            log_file = os.path.join(self.dirs['log_dir'], f"{run_id}.json")
            with open(log_file, "w") as f:
                json.dump(dict(
                    training_config=dataclasses.asdict(tconfig),
                    model_config=dataclasses.asdict(mconfig),
                    data_config=dataclasses.asdict(self.dconfig),
                ), f)

    def finish_training(self, num_epochs, **kwargs):
        print(f"Finished training {self.number_of_models} model(s) for {num_epochs} epochs.")
        losses = {}
        for i, run_id in enumerate(self.run_ids):
            log_file = os.path.join(self.dirs['log_dir'], f"{run_id}.json")
            with open(log_file, "r+") as f:
                log_dict = json.load(f)
                
                # Add losses for this model
                log_dict['best_loss'] = self.best_losses[run_id]
                log_dict['losses'] = self.losses[run_id]
                log_dict['indices'] = self.indices
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
        self.tracked_losses = ["mse"]
        for run_id in self.run_ids:
            loss_weights = self.training_configs[run_id].loss_weights
            if(loss_weights['distribution'] > 0 and "distribution" not in self.tracked_losses):
                self.tracked_losses.append("distribution")
            if(loss_weights['diffusion'] > 0 and "diffusion" not in self.tracked_losses):
                self.tracked_losses.append("diffusion")
                assert 'unet' in kwargs and 'scheduler' in kwargs, "Need to pass in diffusion model"
                self.unet = kwargs['unet']
                self.scheduler = kwargs['scheduler']

    
    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        self.distloss_var_counts = {}
        for run_id in self.run_ids:
            self.distloss_var_counts[run_id] = defaultdict(int)
            self.losses[run_id]['train']['total'] = []
            for loss_type in self.tracked_losses:
                self.losses[run_id]['train'][loss_type] = []
                self.best_losses[run_id][loss_type] = float('inf')
                self.gradients[run_id][loss_type] = {k:[] for k,v in self.models[run_id].named_parameters()}

    def _run_epoch(self, epoch, phase='train'):
        for run_id in self.run_ids:
            self.models[run_id].train(phase=='train')

        """
        Tracking 3 kinds of losses, which can be confusing: 
        batch_losses: losses for a single batch. Because there might be many batches in an epoch, do not save every single batch loss
        int current_losses: instead, divide epoch into batch_logging_interval sized sections, and save the mean of the losses for each section
        int[] epoch_losses: the number of loss items saved for a single epoch is (batches_per_epoch / batch_logging_interval) 
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

    def _run_batch(self, x, y, phase):
        t0 = tru.log_event("run-batch start")
        batch_losses = {}
        for run_id in self.run_ids:
            batch_losses[run_id] = {}
            model, optimizer, tconfig = self.models[run_id], self.optimizers[run_id], self.training_configs[run_id]
            loss_weights = tconfig.loss_weights
            y_hat = model(x)
            mse_loss = self.loss_fn(y_hat, y)
            total_loss = tconfig.loss_weights['mse'] * mse_loss
            batch_losses[run_id]['mse'] = mse_loss.item()
            if('distribution' in self.tracked_losses):
                dist_loss, VAR_IND = self.distribution_loss(y_hat, y, tconfig)
                self.distloss_var_counts[tconfig.run_id][VAR_IND] += 1
                batch_losses[run_id]['distribution'] = dist_loss.item()
                batch_losses[run_id]['VAR_IND'] = VAR_IND
                if(loss_weights['distribution'] > 0):
                    total_loss += loss_weights['distribution'] * dist_loss
                
            if('diffusion' in self.tracked_losses):
                diff_loss = self.diffusion_loss(y_hat, y)
                batch_losses[run_id]['diffusion'] = diff_loss.item()
                if(loss_weights['diffusion'] > 0):  
                    total_loss += loss_weights['diffusion'] * diff_loss
            batch_losses[run_id]['total'] = total_loss.item()
        
            if(phase == 'train'):
                optimizer.zero_grad()
                total_loss.backward()
                if(self.training_configs[run_id].clip_gradients):
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            tru.log_event("run-batch end", duration=time.time() - t0, loss=batch_losses[run_id]['total'], run_id=run_id)
        return(batch_losses)
    
    def distribution_loss(self, y_hat, y, tconfig, epsilon=1e-7):
        VAR_IND = select_distloss_var(tconfig)
        GMM = GaussianMixture(n_components=tconfig.num_gaussians)
        GMM.fit(y[:,VAR_IND].detach().cpu().numpy().reshape(-1, 1))
        mu = torch.tensor(GMM.means_.flatten(), dtype=torch.float64, device=y_hat.device)[None,:]
        pi = torch.tensor(GMM.weights_.flatten(), dtype=torch.float64, device=y_hat.device)[None,:] 
        var = torch.tensor(GMM.covariances_.flatten(), dtype=torch.float64, device=y_hat.device)[None,:]
        pi = torch.clamp(pi, min=epsilon)
        var = torch.clamp(var, min=epsilon)
        
        with torch.no_grad():
            h = 2 * torch.max(var).item()

        bs, n_samples = tconfig.distloss_bs, tconfig.num_distloss_samples
        batch_num = y_hat.shape[0] // bs
        ix, jx = torch.randint(0, batch_num, (n_samples,)), torch.randint(0, batch_num, (n_samples,))
        mask = ix == jx
        while mask.any():
            ix[mask] = torch.randint(0, batch_num, (mask.sum(),))
            jx[mask] = torch.randint(0, batch_num, (mask.sum(),))
            mask = ix == jx
        loss = 0
        for i,j in zip(ix, jx):
            y1 = y_hat[i*bs : (i+1)*bs, VAR_IND]
            y2 = y_hat[j*bs : (j+1)*bs, VAR_IND]
            loss += u_q(y1, y2, mu, var, pi, h).mean()
        return(loss / n_samples, VAR_IND)
    
    def diffusion_loss(self, y_hat, y, T: int, unet, scheduler):
        #raise NotImplementedError("Diffusion loss not implemented")
        # TODO: y and yhat are of size (B, 128) and somehow need to convert into images for diffusion loss
        y, y_hat = construct_image(y), construct_image(y_hat)
        def encode(sample):
            eps = torch.randn(sample.shape, device=device) # BS x C x H x W
            xt = scheduler.add_noise(sample, eps, torch.LongTensor([T])) # noisy image
            return(xt)
        def decode(xt):
            for t in range(T, 0, -1):
                with torch.no_grad():
                    eps_theta = unet(xt, t).sample
                xt = scheduler.step(eps_theta, t, xt).prev_sample
            return(xt)
        y_hat = decode(encode(y_hat))
        return(self.loss_fn(y, y_hat))

    def log_step(self, epoch_losses, batch_losses, step, phase):
        print(f"log stepping at {step}")
        for run_id in self.run_ids:
            for loss_type in self.tracked_losses:
                if(step == 0):
                    epoch_losses[run_id]['distloss_var'] = []
                    epoch_losses[run_id][loss_type] = []
                if(step % self.bli == 0):
                    epoch_losses[run_id]['distloss_var'].append(batch_losses[run_id]['VAR_IND'])
                    epoch_losses[run_id][loss_type].append(batch_losses[run_id][loss_type])
        return(epoch_losses)

    def _log_epoch_info(self, epoch_stats, epoch_num, phase='train'):
        logs_per_epoch = self.batches_per_epoch[phase] // self.bli
        print("\n\n")
        for run_id in self.run_ids:
            for loss_type in self.tracked_losses:
                avg_loss_value = sum(epoch_stats[run_id][loss_type]) / len(epoch_stats[run_id][loss_type])
                self.losses[run_id][phase][loss_type] = self.losses[run_id][phase][loss_type] + epoch_stats[run_id][loss_type]
                print(f"avg {loss_type} loss for epoch {epoch_num}, run_id {run_id}: {avg_loss_value}")
                if(loss_type == 'total' and avg_loss_value < self.best_losses[run_id][loss_type] and phase == 'train'):
                    print(f"saving new checkpoint at epoch {epoch_num}")
                    self.best_losses[run_id][loss_type] = avg_loss_value
                    self._save_checkpoint(run_id)
    def _log_gradients(self):
        pass
                    

class VAETrainer(AbstractTrainer):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank=0):
        """
        Trainer Class for VAE Training
        """
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank)

    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        for run_id in self.run_ids:
            self.losses[run_id]["train"] = {'mse':[], 'kl':[]}
            self.losses[run_id]["eval"] = {'mse':[], 'kl':[]}
            self.best_losses[run_id] = float('inf')
            if(self.training_configs[run_id].log_gradients):
                self.log_gradients = True
                grad_dict = {k:[] for k,_ in self.models[run_id].named_parameters()}
                self.gradients[run_id] = dict(kl_gradients=grad_dict, mse_gradients = grad_dict)

    def _run_batch(self, batch, phase):
        batch_info = {}
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
            batch_info[run_id] = [x_hat, mse.item(), kl_div.item()]

        return(batch_info)

    def _run_train_epoch(self, epoch, phase='train'):
        epoch_info = {run_id : dict(mse=0.0, kl=0.0) for run_id in self.run_ids}
        self.model.train(True)
        dataloader = self.dataloaders[phase]
        with torch.set_grad_enabled(True):
            if(self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(dataloader):
                batch_info = self._run_batch(batch, 'train')
                epoch_info = self._log_step(epoch_info, batch_info, step)
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

    def _log_step(self, epoch_info, batch_info, step, phase='train'):
        for run_id in self.run_ids:
            batch_mse, batch_kl = batch_info[run_id][1], batch_info[run_id][2]
            epoch_info[run_id]['mse'] += batch_mse
            epoch_info[run_id]['kl'] += batch_kl
            if((step+1) % self.bli == 0):
                self.losses[run_id][phase]['mse'].append( epoch_info[run_id]['mse'] / self.bli )
                self.losses[run_id][phase]['kl'].append( epoch_info[run_id]['kl'] / self.bli )
                epoch_info[run_id] = dict(mse=0.0, kl=0.0)
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




def score_function(x, mu, var, pi):
    log_probs = torch.log(pi) - 0.5 * torch.log(var * 2 * torch.pi) - (x[:,None] - mu) ** 2 / (2 * var)
    qx = torch.exp(torch.logsumexp(log_probs, dim=1, keepdim=True))
        # Compute weighted derivative terms: pi * N(x|mu, var) * (x - mu) / var
    weighted_terms = torch.exp(log_probs) * (x[:,None] - mu) / var  # (N, K)
    # Sum over components and divide by q(x)
    score = -weighted_terms.sum(dim=1, keepdim=True) / qx    # (N, 1)
    return score.squeeze()


def u_q(x_1, x_2, mu, var, pi, h):
    assert len(x_1.shape) == 1 and len(x_2.shape) == 1, "expected 1d input"
    # x_1 and x_2 are samples from q(x), compared to knowledge distribution p(x) represented by mixture model. Thus, they are of shape (n_samples, d)
    sq1 = score_function(x_1, mu, var, pi)
    sq2 = score_function(x_2, mu, var, pi)

    # Compute the final result based on bandwidth h
    if h == float('inf'):
        return torch.outer(sq1, sq2)
    
    diffs = x_1.unsqueeze(1) - x_2.unsqueeze(0)
    res = torch.outer(sq1, sq2)
    res += (sq1.reshape(-1,1) * diffs) / h**2
    res -= (sq2.reshape(1, -1) * diffs)/h**2
    res += (h**-2 - h**-4 * diffs ** 2)
    res = res * torch.exp(- (diffs ** 2) / (2 * h ** 2)) # kernel matrix
    return(res)

def select_distloss_var(tconfig, **kwargs):
    if(isinstance(tconfig.distloss_var_inds, int)):
        return(tconfig.distloss_var_inds)
    elif(len(tconfig.distloss_var_inds) == 1):
        return(tconfig.distloss_var_inds[0])

    match tconfig.distloss_var_sel.lower():
        case "uniform":
            return(random.choice(tconfig.distloss_var_inds))
        case _:
            return(random.choice(tconfig.distloss_var_inds))
    
    


def distribution_loss(y_hat, y, tconfig, h=None):
    GMM = GaussianMixture(n_components=tconfig.num_gaussians)
    GMM.fit(y[:,tconfig.distloss_var_ind].detach().cpu().numpy().reshape(-1, 1))
    mu = torch.tensor(GMM.means_.flatten(), dtype=torch.float64, device=y_hat.device)[None,:] # (n_components,) (nc, d=1 flattened)
    pi = torch.tensor(GMM.weights_.flatten(), dtype=torch.float64, device=y_hat.device)[None,:] # (n_components,)
    var = torch.tensor(GMM.covariances_.flatten(), dtype=torch.float64, device=device)[None,:] # (n_components,) (technically (n_c, d,d) but flattened), 
    epsilon = 1e-7 # Ensure pi and var are positive and non-zero for stability
    pi = torch.clamp(pi, min=epsilon)
    var = torch.clamp(var, min=epsilon)

    if(h is None):
        with torch.no_grad():
            h = 2 * torch.max(var).item()

    bs, n_samples = tconfig.distloss_bs, tconfig.num_distloss_samples
    batch_num = y_hat.shape[0] // bs
    ix, jx = torch.randint(0, batch_num, (n_samples,)), torch.randint(0, batch_num, (n_samples,))
    mask = ix == jxy_hat.sh
    while mask.any():
        ix[mask] = torch.randint(0, batch_num, (mask.sum(),))
        jx[mask] = torch.I(0, batch_num, (mask.sum(),))
        mask = ix == jx
    loss = 0
    for i,j in zip(ix, jx):
        y1 = y_hat[i*bs : (i+1)*bs, tconfig.distloss_var_ind]
        y2 = y_hat[j*bs : (j+1)*bs, tconfig.distloss_var_ind]
        loss += u_q(y1, y2, mu, var, pi, h).mean()
    return(loss)
    
            