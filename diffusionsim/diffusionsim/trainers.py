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
import random
import wandb
from sklearn.mixture import GaussianMixture

class AbstractTrainer(ABC):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank, indices=None, **kwargs):
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        self.rank = rank
        self.phases = tconfigs[0].phases
        assert isinstance(mconfigs, list) and isinstance(tconfigs, list) and len(mconfigs) == len(tconfigs), "mconfigs and tconfigs must be lists of same length"
        self.dconfig = dconfig
        dls, self.indices = tru.load_dataloaders(dconfig, log=True, indices=indices)
        self.dutils = dls[0].dataset.dutils
        self.dataloaders = {}
        for i, phase in enumerate(self.phases):
            self.dataloaders[phase] = dls[i]
        
        self.number_of_models = len(mconfigs)
        self.base_run_id = base_run_id
        self.run_ids = [tconfig.run_id for tconfig in tconfigs]
        self.mconfigs = dict(zip(self.run_ids, mconfigs))
        self.training_configs = dict(zip(self.run_ids, tconfigs))
        self.distributed = bool(tconfigs[0].distributed_training)
        self.models = dict(zip(self.run_ids, [tru.load_model(mconfig, device=self.device, distributed=self.distributed) for mconfig in mconfigs]))
        optimizers = [tru.create_optimizer(model, tconfig) for model, tconfig in zip(self.models.values(), tconfigs)]
        self.optimizers = dict(zip(self.run_ids, optimizers))
        self.lr_schedulers = dict(zip(self.run_ids, [tru.load_lr_scheduler(tconfig, optim, dls[0]) for tconfig, optim in zip(tconfigs, optimizers)]))
        self.loss_fn = loss_fn
        brid = self.base_run_id[:-1] if self.base_run_id[-1].isdigit() else self.base_run_id
        self.dirs = dict(
            log_dir = base_dir,
            ckpt_dir = os.path.join(base_dir, "checkpoints", brid),
            output_dir = os.path.join(base_dir, "outputs", brid),
        )
        
        self.log_file_path = os.path.join(self.dirs['log_dir'], f"{self.base_run_id}.json")
        self.bli = tconfigs[0].batch_logging_interval
        self.ckpt_interval = tconfigs[0].batch_checkpoint_interval
        self.restarted = False
        self.losses, self.gradients, self.best_losses, self.nan_incidents = {}, {}, {}, {}
        self.batches_per_epoch = dict(zip(self.phases, [len(self.dataloaders[phase]) for phase in self.phases]))
    
    def setup_training(self, num_epochs, log=True, save_indices=False):
        self.current_epoch = 0
        self.log = log
        for phase in self.phases:
            self.dataloaders[phase].dataset.log = log
        print(f"Getting ready to train {self.number_of_models} model(s) for {num_epochs} epochs on device {self.device}; configs at {self.log_file_path}", flush=True)
        for dirname in self.dirs.values():
            Path(dirname).mkdir(parents=True, exist_ok=True)
        self.losses, self.gradients, self.best_losses, self.nan_incidents = {}, {}, {}, {}
        self.batches_per_epoch = dict(zip(self.phases, [len(self.dataloaders[phase]) for phase in self.phases]))
        master_dict = {'data_config': dataclasses.asdict(self.dconfig)}
        if (self.dconfig.shuffle_indices and save_indices):
            # only need to save indices if were shuffled
            master_dict['indices'] = [i.tolist() for i in self.indices]
        for run_id in self.run_ids:
            tconfig = self.training_configs[run_id]
            tconfig.num_epochs = num_epochs
            if(tconfig.log_gradients):
                self.log_gradients = True
            master_dict[run_id] = dict(
                    training_config=dataclasses.asdict(tconfig),
                    model_config=dataclasses.asdict(self.mconfigs[run_id]),
            )
        
        self.run = wandb.init(
            entity="samart-agr-columbia-university", 
            project="diffusionsim",
            name=self.base_run_id,
            config=master_dict,
        )
        
        print(f"Saving configs to {self.log_file_path}", flush=True)
        with open(self.log_file_path, "w") as f:
            json.dump(master_dict, f)
        

    def _save_checkpoint(self, run_id, cid='', **kwargs):
        model = self.models[run_id]
        ckp = model.module.state_dict() if self.distributed else model.state_dict()
        ckp = {k: v.detach().cpu() for k, v in ckp.items()}
        path = os.path.join(self.dirs['ckpt_dir'], f"{cid}{run_id}-ckpt.pt")
        torch.save(ckp, path)
        print(f"Model saved at {path}")
      
    def restart_from_ckpt(self, num_epochs, log, cid='', phase='train'):
        self.setup_training(num_epochs, log=log)
        self.restarted = True
        with open(self.log_file_path, 'r') as f:
            master_dict = json.load(f)
        for run_id in self.run_ids:
            ckpt_path = os.path.join(self.dirs['ckpt_dir'], f"{cid}{run_id}-ckpt.pt")
            if(os.path.exists(ckpt_path)):
                print(f"Loading checkpoint from {ckpt_path}")
                self.models[run_id].load_state_dict(torch.load(ckpt_path, map_location=self.device))
                #self.optimizers[run_id].load_state_dict(torch.load(ckpt_path))
                #self.lr_schedulers[run_id].load_state_dict(torch.load(ckpt_path))
            if('losses' in master_dict[run_id]):
                self.losses.setdefault(run_id, {}).update(master_dict[run_id]['losses'])
            if('gradients' in master_dict[run_id]):
                self.gradients.setdefault(run_id, {}).update(master_dict[run_id]['gradients'])
        if(phase == 'train'):
            self.train(num_epochs, log=log)
        else:
            self.evaluate()
    
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

    def evaluate(self, log=True):
        self.log = log
        assert len(self.phases) > 1, "Need to define more than one phase to evaluate"
        phase = "eval"
        self.distloss_var_counts = {}
        eval_stats = self._run_epoch(0, phase)
        self._log_epoch_info(eval_stats, 0, phase)

    def train(self, num_epochs, log=True):
        if not self.restarted:
            self.setup_training(num_epochs, log)
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch}/{num_epochs-1}", flush=True)
            print("_" * 10, flush=True)
            stats = self._run_epoch(epoch)
            if(stats is None):
                print(f"Epoch {epoch} is bad; stopping", flush=True)
                break
            self._log_epoch_info(stats, epoch, phase="train")
        
        self.finish_training(num_epochs)

    def finish_training(self, num_epochs, **kwargs):
        print(f"Finished training {self.number_of_models} model(s) for {num_epochs} epochs.", flush=True)
        losses = {}
        with open(self.log_file_path, "r") as f:
            master_dict = json.load(f)
        for i, run_id in enumerate(self.run_ids):
            log_dict = master_dict[run_id]  
            # Add losses for this model
            log_dict['losses'] = self.losses[run_id]
            log_dict['nan_incidents'] = self.nan_incidents.get(run_id)
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
                log_dict['gradients'] = self.gradients.setdefault(run_id, {})
            master_dict[run_id] = log_dict
        
        with open(self.log_file_path, "w") as f:
            json.dump(master_dict, f)
                    
        return 0

class ClimsimTrainer(AbstractTrainer):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank=0, indices=None, **kwargs):
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank, indices)
        self.tracked_losses = ["mse"]
        for run_id in self.run_ids:
            loss_weights = self.training_configs[run_id].loss_weights
            if(loss_weights['distribution'] > 0 and "distribution" not in self.tracked_losses):
                self.tracked_losses.append("distribution")
            if(loss_weights['diffusion'] > 0 and "diffusion" not in self.tracked_losses):
                self.tracked_losses.append("diffusion")
        
        if(len(self.tracked_losses) > 1):
            self.tracked_losses.append("total")

        if ("diffusion" in self.tracked_losses):
            self.unet = kwargs['unet'] if('unet' in kwargs) else tru.load_diffusion_model().to(self.device)
            scheds = [kwargs['scheduler'] for _ in range(len(self.run_ids))] if 'scheduler' in kwargs else [tru.load_scheduler(mconfig) for mconfig in self.mconfigs.values()]
            self.schedulers = dict(zip(self.run_ids, scheds))
        #if ("distribution" in self.tracked_losses):
        

    def _make_image(self, y, image_dim=2):
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        ds = self.dataloaders['train'].dataset
        # desired output is (BS, C, H, W)
        return(tru.imagify(y, ds.dutils, 'y', image_dim))

    def setup_training(self, num_epochs, log=True):
        super().setup_training(num_epochs, log)
        self.distloss_var_counts = {}
    
    def _return_loss_weights(self, run_id):
        tconfig = self.training_configs[run_id]
        lws = {}
        for loss_type, loss_sched in tconfig.loss_schedule.items():
            lws[loss_type] = tconfig.loss_weights[loss_type]
            if(isinstance(loss_sched, list) and len(loss_sched) == 2):
                if not (loss_sched[0] <= self.current_epoch < loss_sched[1]):
                    lws[loss_type] = 0
        return(lws)
    
    def finish_training(self, num_epochs):
        super().finish_training(num_epochs, distloss_var_counts=self.distloss_var_counts)

    def _run_epoch(self, epoch, phase='train'):
        for run_id in self.run_ids:
            self.models[run_id].train(phase=='train')

        """
        Tracking 3 kinds of losses, which can be confusing: 
        batch_losses: losses for a single batch. Because there might be many batches in an epoch, do not save every single batch loss
        int current_losses: instead, divide epoch into batch_logging_interval sized sections, and save the mean of the losses for each section
        int[] epoch_losses: the number of loss items saved for a single epoch is (batches_per_epoch / batch_logging_interval) 
        """
        e0 = tru.log_event(f"epoch-{epoch} start")
        epoch_losses = defaultdict(dict)
        with torch.set_grad_enabled(phase=='train'):
            if(self.distributed):
                self.dataloaders[phase].sampler.set_epoch(epoch)
            for step, (X, Y) in enumerate(self.dataloaders[phase]):
                if(step % 100 == 0):
                    print(f"Currently at epoch {epoch}, step {step}/{len(self.dataloaders[phase])}", flush=True)                
                batch_losses = self._run_batch(X.to(self.device), Y.to(self.device), phase, step)
                if(batch_losses is None):
                    print(f"Batch {step} is bad; stopping", flush=True)
                    break
                self.log_step(epoch_losses, batch_losses, step, phase)
                
        tru.log_event(f"epoch-{epoch} end", duration=time.time() - e0)
        return(epoch_losses)

    def _run_batch(self, x, y, phase, step):
        if(self.log):
            t0 = tru.log_event("run-batch start", step=step)
        batch_losses = {}
        for run_id in self.run_ids:
            batch_losses[run_id] = {}
            model, optimizer, tconfig = self.models[run_id], self.optimizers[run_id], self.training_configs[run_id]
            lws = self._return_loss_weights(run_id)
            y_hat = model(x)
            mse_loss = self.loss_fn(y_hat, y)
            if(phase == 'train' and self._gradients_ops(run_id, "mse", mse_loss, step)):
                self.nan_incidents[run_id][-1]['inputs'] = (x, y)
            total_loss = lws['mse'] * mse_loss
            batch_losses[run_id]['mse'] = mse_loss.item()
            if('distribution' in self.tracked_losses):
                dist_loss, VAR_IND = self.distribution_loss(y_hat, y, tconfig, step)
                if(phase == 'train' and self._gradients_ops(run_id, "distribution", dist_loss, step)):
                    self.nan_incidents[run_id][-1]['inputs'] = (x, y)
                self.distloss_var_counts.setdefault(run_id, defaultdict(int))[VAR_IND] += 1
                batch_losses[run_id]['distribution'] = dist_loss.item()
                batch_losses[run_id]['VAR_IND'] = VAR_IND
                if(lws['distribution'] > 0 and not torch.isnan(dist_loss)):
                    total_loss += lws['distribution'] * dist_loss
                
            if('diffusion' in self.tracked_losses):
                diff_loss = self.diffusion_loss(y_hat, run_id)
                if(phase == 'train' and self._gradients_ops(run_id, "diffusion", diff_loss, step)):
                    self.nan_incidents[run_id][-1]['inputs'] = (x, y)
                batch_losses[run_id]['diffusion'] = diff_loss.item()
                if(lws['diffusion'] > 0 and not torch.isnan(diff_loss)):  
                    total_loss += lws['diffusion'] * diff_loss
            
            batch_losses[run_id]['total'] = total_loss.item()
            if(phase == 'train'):
                optimizer.zero_grad()
                total_loss.backward(retain_graph=False)
                if(self.training_configs[run_id].clip_gradients):
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if(self.lr_schedulers[run_id] is not None):
                    self.lr_schedulers[run_id].step()
        losses = [batch_losses[run_id]['total'] for run_id in self.run_ids]
        if(self.log):
            tru.log_event("run-batch end", duration=time.time() - t0, loss=losses, step=step)
        return(batch_losses)
    
    def distribution_loss(self, y_hat, y, tconfig, step, epsilon=1e-3):
        VAR_IND, num_gaussian = select_distloss_var(tconfig, step)
        assert isinstance(num_gaussian, int) and isinstance(VAR_IND, int), "num_gaussian and VAR_IND must be integers"
        GMM = GaussianMixture(n_components=num_gaussian, reg_covar=epsilon)
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
        
    def diffusion_loss(self, y_hat, run_id):
        #raise NotImplementedError("Diffusion loss not implemented")
        # TODO: y and yhat are of size (B, 128) and somehow need to convert into images for diffusion loss
        tconfig = self.training_configs[run_id]
        scheduler = self.schedulers[run_id]
        y_image = self._make_image(y_hat, image_dim=2)
        def encode(sample):
            eps = torch.randn(sample.shape, device=sample.device) # BS x C x H x W
            xt = scheduler.add_noise(sample, eps, torch.LongTensor([tconfig.diffusion_loss_noise_level])) # noisy image
            return(xt)
        def decode(xt):
            with torch.no_grad():
                for t in range(tconfig.diffusion_loss_noise_level, 0, -tconfig.diffusion_loss_decoding_interval):
                    eps_theta = self.unet(xt, t).sample
                    xt = scheduler.step(eps_theta, t, xt).prev_sample
            return(xt)
        y_image_denoised = decode(encode(y_image))
        return(self.loss_fn(y_image_denoised, y_image))

    def log_step(self, epoch_losses, batch_losses, step, phase, wandb=True):
        if(wandb):
            wandb_metrics = {"epoch" : self.current_epoch}
            for run_id in self.run_ids:
                if('distribution' in self.tracked_losses):
                    wandb_metrics[f"{phase}/{run_id}/distloss_var"] = batch_losses[run_id]['VAR_IND']
                for loss_type in self.tracked_losses:
                    wandb_metrics[f"{phase}/{run_id}/{loss_type}"] = batch_losses[run_id][loss_type]
            #global_step = self.current_epoch * len(self.dataloaders[phase]) + step
            self.run.log(wandb_metrics)
        
        if (step % self.bli == 0):
            #with open(self.log_file_path, "r") as f:
            #    master_dict = json.load(f)
            for run_id in self.run_ids:
                if('distribution' in self.tracked_losses):
                    epoch_losses[run_id].setdefault('distloss_var', []).append(batch_losses[run_id]['VAR_IND'])
                for loss_type in self.tracked_losses:
                    epoch_losses[run_id].setdefault(loss_type, []).append(batch_losses[run_id][loss_type])
            #    master_dict[run_id][f'{phase}_epoch_losses'] = epoch_losses[run_id]
            #with open(self.log_file_path, "w") as f:
            #    json.dump(master_dict, f)
        
        if ( (step + 1) % self.ckpt_interval == 0 and phase == 'train'): 
            for run_id in self.run_ids:
                self._save_checkpoint(run_id)

    def _log_epoch_info(self, epoch_stats, epoch_num, phase='train'):
        with open(self.log_file_path, "r") as f:
            master_dict = json.load(f)
        print("\n\n", flush=True)
        for run_id in self.run_ids:
            ldict = self.losses.setdefault(run_id, {}).setdefault(phase, {})
            for loss_type in self.tracked_losses:
                ldict[loss_type] = ldict.setdefault(loss_type, []) + epoch_stats[run_id][loss_type]
                avg_loss = sum(epoch_stats[run_id][loss_type]) / len(epoch_stats[run_id][loss_type])
                print(f"avg {loss_type} loss for epoch {epoch_num}, run_id {run_id}: {avg_loss}", flush=True)
                if(phase == 'train' and loss_type == 'total'):
                    if(avg_loss < self.best_losses.setdefault(run_id, float('inf'))):
                        print(f"new best! saving new checkpoint at epoch {epoch_num}", flush=True)
                        self.best_losses[run_id] = avg_loss
                        self._save_checkpoint(run_id, cid='best-')
            
                master_dict[run_id].setdefault('losses', {}).setdefault(phase, {}).setdefault(loss_type, []).extend(ldict[loss_type])
                
                if(phase == 'train'):
                    master_dict[run_id]['gradients'] = self.gradients[run_id]
            master_dict[run_id][f'{phase}_epoch_losses'] = None

        with open(self.log_file_path, "w") as f:
            json.dump(master_dict, f)
    
    def _gradients_ops(self, run_id, loss_type, loss, step):
        #assert loss_type in self.tracked_losses, f"loss_type {loss_type} must be one of {self.tracked_losses}"
        model, optimizer = self.models[run_id], self.optimizers[run_id]

        def register_incident(self, msg, tensor_name=None, tensor=None):
            print(f"[NaN-Detect] step={step}  {run_id=} {loss_type=}  >>> {msg}", flush=True)
            payload = dict(step=step, run_id=run_id, loss_type=loss_type, msg=msg, 
                           tensor_name=tensor_name, 
                           tensor_sample=tensor.clone().cpu() if tensor is not None else None, 
                           state_dict={k: v.clone().cpu() for k, v in model.state_dict().items()}
            )
            self.nan_incidents.setdefault(run_id, []).append(payload)
        
        if not torch.isfinite(loss):
            register_incident("LOSS is not finite (NaN or Inf)", "loss", loss.detach())
            return -1
        """
        # ------------------------------------------------------------------ #
        # 2. Check *parameter values* BEFORE backward (stage d in the table)
        # ------------------------------------------------------------------ #
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():
                register_incident("PARAM has NaN/Inf BEFORE backward", n, p)

        # ------------------------------------------------------------------ #
        # 3. Probe gradients for THIS loss only (retain_graph=True)
        # ------------------------------------------------------------------ #
        
        optimizer.zero_grad(set_to_none=True)           # clean slate
        loss.backward(retain_graph=True)

        
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                register_incident("GRAD has NaN/Inf", n, p.grad)

        """
        if (self.training_configs[run_id].log_gradients and step % self.bli == 0):
            optimizer.zero_grad(set_to_none=True)           # clean slate
            loss.backward(retain_graph=True)
            gdict = self.gradients.setdefault(run_id, {}).setdefault(loss_type, {})
            for n, p in model.named_parameters():
                if p.grad is not None and torch.isfinite(p.grad).all():
                    gdict.setdefault(n, []).append(
                        (p.grad.mean().item(), p.grad.std().item())
                    )
                elif(not torch.isfinite(p.grad).all()):
                    register_incident("GRAD has NaN/Inf", n, p.grad)
                    return -1

        # ------------------------------------------------------------------ #
        # 5. Clean up so real backward sees a fresh gradient buffer
        # ------------------------------------------------------------------ #
        optimizer.zero_grad(set_to_none=True)

        # Let caller know if we saw anything suspicious
        return 0

class DiffusionTrainer(AbstractTrainer):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank=0, indices=None, **kwargs):
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank, indices)
        self.scheduler = tru.load_scheduler(mconfigs[0])
    
    def _run_batch(self, images, phase='train'):
        if(self.log):
            t0 = tru.log_event("run-batch start")
        batch_losses = {}
        for run_id in self.run_ids:
            model, optimizer = self.models[run_id], self.optimizers[run_id]
            tconfig, lr_scheduler = self.training_configs[run_id], self.lr_schedulers[run_id]
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
                if(lr_scheduler is not None):
                    lr_scheduler.step()
            batch_losses[run_id] = loss.item()
        if(self.log):
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
                self.log_step(epoch, step, batch_losses, phase, epoch_losses)
                tru.log_event(f"batch-{step} end", batch=step, duration= time.time() - tt0)
                if(step % 50 == 0):
                    print(f"Currently at epoch {epoch}, step {step}", flush=True)
        return(epoch_losses)

    def log_step(self, epoch, step, batch_losses, phase, epoch_losses):
        for run_id in self.run_ids:
            if(step % self.bli == 0 or step == len(self.dataloaders[phase])):
                epoch_losses[run_id].append(batch_losses[run_id])

        if ((step+1) % self.ckpt_interval == 0):
            print(f"epoch {epoch}, step {step}: saving checkpoint", flush=True)
            for run_id in self.run_ids:
                self._save_checkpoint(run_id)
        return(epoch_losses)
    
    def _log_epoch_info(self, epoch_num, epoch_losses, phase='train'):
        for run_id in self.run_ids:
            ld = self.losses.setdefault(run_id, {})
            ld.setdefault(phase, []).extend(epoch_losses[run_id])
            avg_loss = sum(epoch_losses[run_id]) / len(epoch_losses[run_id])
            if(avg_loss < self.best_losses.setdefault(run_id, float('inf')) and self.training_configs[run_id].save_best_epoch):
                self.best_losses[run_id] = avg_loss
                self._save_checkpoint(run_id, cid='best-')
        
    #def _save_checkpoint(self, epoch, cid):
        # extend such that training can be resumed from checkpoint alone. 
        # possibly create and delete as part of finish training? 
    #    super()._save_checkpoint(epoch,  cid=cid)
     
    def train(self, num_epochs=0, log=True):
        self.setup_training(num_epochs, log)
        print(f"Training for {num_epochs} epochs", flush=True)
        try: 
            for epoch in range(num_epochs): # set in setup_training
                e0 = tru.log_event("epoch start", epoch=epoch)
                loss = self._run_epoch(epoch)
                self._log_epoch_info(epoch, loss, phase='train')
                tru.log_event("epoch end", epoch=epoch, duration=time.time() - e0, epoch_loss=loss)
        except Exception as e:
            print(f"Some error occurred during training: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
        self.finish_training(num_epochs)
    
    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        self.event_file = 0 #open(os.path.join(self.dirs['output_dir'], "event_log.txt"), "a")
        self.noise_timesteps = {}

    def finish_training(self, num_epochs, **kwargs):
        # to do: add github hash as well so can reproduce code base at time of training run
        noise_timesteps = {}
        for run_id in self.run_ids:
            noise_timesteps[run_id] = self.noise_timesteps[run_id].numpy().tolist()
        super().finish_training(num_epochs, sampled_timesteps=noise_timesteps)
        #if(self.event_file):
        #    self.event_file.close()

def score_function(x, mu, var, pi, clamp_val=10, epsilon=1e-7):
    mahal = (x[:,None] - mu) ** 2 / (2 * var)
    log_probs = torch.log(pi) - 0.5 * torch.log(var * 2 * torch.pi) - torch.clamp(mahal, max=clamp_val)
    qx = torch.exp(torch.logsumexp(log_probs, dim=1, keepdim=True)) + epsilon
        # Compute weighted derivative terms: pi * N(x|mu, var) * (x - mu) / var
    weighted_terms = torch.exp(log_probs) * torch.clamp((x[:,None] - mu) / var, max=clamp_val, min=-clamp_val)  # (N, K)
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
    res = res * torch.exp(- (diffs ** 2) / (2 * h ** 2)) # kernel 
    return(torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4))

def select_distloss_var(tconfig, step, **kwargs):
    var_inds = tconfig.distloss_var_inds
    ng = tconfig.num_gaussians
    if(isinstance(var_inds, int)):
        assert isinstance(ng, int), "num_gaussians must be integer if distloss_var_inds is integer"
        return(var_inds, ng)
    assert isinstance(var_inds, list), "distloss_var_inds must be an integer or list"
    assert isinstance(ng, list) and len(var_inds) == len(ng), "num_gaussians must be list of same length as distloss_var_inds"
    if(len(var_inds) == 1):
        i = 0
    elif(tconfig.distloss_var_sel.lower() == "uniform"):
        i = random.randrange(len(var_inds))
    elif(tconfig.distloss_var_sel.lower() == "cycle"):
        i = step % len(var_inds)
    else:
        i = random.randrange(len(var_inds))
    return(var_inds[i], ng[i])
    
                    
class VAETrainer(AbstractTrainer):
    def __init__(self, dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank=0, indices=None, **kwargs):
        """
        Trainer Class for VAE Training
        """
        super().__init__(dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id, rank, indices)

    def setup_training(self, num_epochs):
        super().setup_training(num_epochs)
        for run_id in self.run_ids:
            self.losses[run_id]["train"] = {'mse':[], 'kl':[]}
            self.losses[run_id]["eval"] = {'mse':[], 'kl':[]}
            self.best_losses[run_id] = float('inf')
            if(self.training_configs[run_id].log_gradients):
                self.log_gradients = True

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
                    gdict = self.gradients.setdefault(run_id, {})
                    # Compute and store gradients for MSE
                    mse.backward(retain_graph=True)
                    for name, p in model.named_parameters():
                        gdict.setdefault('mse_gradients', defaultdict(list))[name].append(p.grad.mean().item())
                    optimizer.zero_grad()  # Clear the gradients again
    
                    # Compute and store gradients for KL divergence
                    (tconfig.beta*kl_div).backward(retain_graph=True)
                    if(tconfig.log_gradients):
                        for name, p in model.encoder.named_parameters():
                            name = "encoder." + name
                        gdict.setdefault('kl_gradients', defaultdict(list))[name].append(p.grad.mean().item())
                    optimizer.zero_grad()  # Clear the gradients again

            # Combine the loss and backpropagate
                total_loss = mse + tconfig.beta * kl_div
                total_loss.backward()
                optimizer.step()
                if(self.lr_schedulers[run_id] is not None):
                    self.lr_schedulers[run_id].step()
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
            print(f"{phase} stats [mse : {epoch_mse}; kl : {epoch_kl}]", flush=True)
            is_best_loss = (epoch_mse+epoch_kl)<self.best_loss
            if(is_best_loss and self.rank == 0 and phase == 'train'):
                print(f"saving new checkpoint at epoch {epoch_num}", flush=True)
                self.best_loss = epoch_mse + epoch_kl
                self._save_checkpoint(epoch_num, cid='best-') # TODO: I dont think this is running

    def _log_step(self, epoch_info, batch_info, step, phase='train'):
        for run_id in self.run_ids:
            batch_mse, batch_kl = batch_info[run_id][1], batch_info[run_id][2]
            epoch_info[run_id]['mse'] += batch_mse
            epoch_info[run_id]['kl'] += batch_kl
            if((step+1) % self.bli == 0):
                self.losses[run_id][phase]['mse'].append( epoch_info[run_id]['mse'] / self.bli )
                self.losses[run_id][phase]['kl'].append( epoch_info[run_id]['kl'] / self.bli )
                epoch_info[run_id] = dict(mse=0.0, kl=0.0)
                print(f"Batch {step}/{self.batches_per_epoch} [mse: {mse}  kl: {kl}]", flush=True)
                if(self.training_configs[self.run_ids[0]].log_gradients and False):
                    for name, p in self.model.named_parameters():
                        gnorm = torch.linalg.norm(p.grad)
                        self.gradients[name].append(gnorm.detach().item())

    def _run_epoch(self, epoch, phase='train'):
        if(phase == 'train'):
            self._run_train_epoch(epoch)
        else:
            self._run_eval_epoch(epoch)
        print(f"Eval Loss on epoch {epoch}: [mse : {self.eval_losses['mse'][-1]}, kl {self.eval_losses['kl'][-1]}]", flush=True)
