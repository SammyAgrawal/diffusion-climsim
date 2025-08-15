import os
import sys
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
from tqdm import tqdm
from pathlib import Path
import re
import time
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim.trainers import DiffusionTrainer
import torch
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

rank = 0
device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

REF_BATCH_SIZE = 128
dataset_type = "xbatch"
in_notebook = True


def setup_diffusion_run(num_models, exp_id, base_run_id, data_vars='v1', batch_size=128):
    dconfig = tru.my_dconfig(source = "local-vzarr", data_vars=data_vars, in_notebook=in_notebook, dataset_type=dataset_type, batch_size=batch_size, use_tendencies=True)
    tconfigs, mconfigs = [], []

    learning_rates = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
    unet_layers_per_block = [1, 1, 2, 2, 1]

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=f"{base_run_id}{lettering[i]}")
        tconfig.phases = ['train']
        tconfig.learning_rate = 1e-5 * batch_size / REF_BATCH_SIZE
        tconfig.lr_scheduler = "cosine"
        tconfig.lr_warmup_steps = 200
        tconfig.max_T_sample = 51
        tconfig.clip_gradients = True

        unet = tru.UNetParams()
        if(i == 0):
            unet.block_out_channels = (128, 256, 512) if data_vars == "v1" else (256, 512, 1024)
            unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
            unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
        elif(i % 2 == 0):
            unet.block_out_channels = (128, 256, 256, 512) if data_vars == "v1" else (256, 512, 512, 1024)
            unet.down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            unet.up_block_types = ("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D")
        else:
            unet.block_out_channels = (128, 256, 256, 512) if data_vars == "v1" else (256, 512, 512, 1024)
            unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D")
            unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        unet.in_channels = 128 if data_vars == "v1" else 368
        unet.out_channels = unet.in_channels
        unet.layers_per_block = unet_layers_per_block[i]
        unet.norm_num_groups = 2

        scheduler = tru.SchedulerParams()
        scheduler.beta_schedule = "linear"
        scheduler.clip_sample = False
        scheduler.clip_sample_range = 4.0
        scheduler.beta_end = 0.02

        mconfig = tru.ModelConfig(unet=unet, scheduler=scheduler)
        mconfig.model_type = 'ddom_diffusion'
        mconfig.data_vars = data_vars

        tconfigs.append(tconfig)
        mconfigs.append(mconfig)
    return(tconfigs, mconfigs, dconfig)


if __name__ == "__main__":
    num_models = 4
    exp_id = "full_vars_diffusion"
    run_id = "test_attn"
    base_dir = os.path.join("/mnt/home/ssa2206/Climsim/experiments", exp_id)
    tconfigs, mconfigs, dconfig = setup_diffusion_run(num_models, exp_id, run_id, data_vars = 'v2', batch_size = 256)
    run_start_time = tru.log_event("run start", data_params = tru.asdict(dconfig.dataloader_params))
    t0 =  tru.log_event("setup start", run_id=run_id)
    #model = tru.load_model_from_ckpt("trial_1-ckpt.pt", mconfig, exp_id, EXP_DIR)
    loss_fn = torch.nn.MSELoss()
    trainer = DiffusionTrainer(dconfig, mconfigs, tconfigs, loss_fn, base_dir, base_run_id=run_id, )

    """
    ckpt = os.path.join(base_dir, "checkpoints", "lr-searcha-ckpt.pt")
    weights = torch.load(ckpt, map_location=trainer.device)
    noise_levels = [0.0, 0.5, 1.0, 1.0, 1.5]
    print("Adding noise to model weights")
    for i, run_id in enumerate(trainer.models.keys()):
        model = trainer.models[run_id]
        model.load_state_dict(weights)
        noise_level = noise_levels[i]
        for name, param in model.named_parameters():
            if not param.requires_grad or re.search(r'\.norm\d+\.(weight|bias)', name):
                continue
            with torch.no_grad():
                std = torch.std(param).item()
                #mean = torch.mean(param).item()
                if 'time_embedding' in name:
                    param.add_(0.3 * noise_level * std * torch.randn_like(param))
                elif 'conv' in name:
                    param.add_(0.05 * noise_level * std * torch.randn_like(param))
                elif re.search(r'\.weight|\.bias', name):
                    param.add_(0.5 * noise_level * std * torch.randn_like(param))
                else:
                    print(f"  -> Skipping: {name}")
    """
    tru.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=20, log=True)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)




    def _update_gradnorm_weights(self, run_id, batch_losses):
        params = self.training_configs[run_id].loss_weight_params
        W = self.loss_weights[run_id]
        losses = torch.stack([batch_losses[l].detach() for l in self.tracked_losses])
        l0 = self.L0[run_id]['l0'].detach().clone()
        if "VAR_IND" in batch_losses and batch_losses['VAR_IND'] in self.L0[run_id]:
            i = self.tracked_losses.index("distribution")
            l0[i] = self.L0[run_id][batch_losses['VAR_IND']]
        loss_ratio = losses / l0
        grads = []
        param = self.models[run_id].get_param(params['gradnorm_layer'])
        for i, loss_type in enumerate(self.tracked_losses):
            grad = torch.autograd.grad(W[i] * batch_losses[loss_type], param, retain_graph=True, create_graph=True)[0]
            grads.append(torch.norm(grad))
        grads = torch.stack(grads)
        target_grads = grads.detach().mean() * (loss_ratio / loss_ratio.mean()) ** params['alpha']
        gradnorm_loss = self.gradnorm_loss_fn(grads, target_grads)
        print("loss weights before update\n", self.loss_weights[run_id])
        self.lw_optimizers[run_id].zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.lw_optimizers[run_id].step()
        with torch.no_grad():
            if (W < 0).any():
                W = torch.exp(W)
            self.loss_weights[run_id] = torch.nn.Parameter((params['T'] * W/W.sum()).detach())
            self.lw_optimizers[run_id] = torch.optim.SGD([self.loss_weights[run_id]], lr=params['lr'])
            
        print("loss weights after updat and norm\n", self.loss_weights[run_id])

    def apply_loss(self, y_hat, x, y, batch_losses, run_id, step):
        #model, optimizer, tconfig = self.models[run_id], self.optimizers[run_id], self.training_configs[run_id]
        self.models[run_id].zero_grad()
        VAR_IND = None
        for loss_type in self.tracked_losses:
            if(loss_type == "mse"):
                loss = self.loss_fn(y_hat, y)
            elif(loss_type == "distribution"):
                loss, VAR_IND = distribution_loss(self, y_hat, y, run_id, step)
                batch_losses[run_id]['VAR_IND'] = VAR_IND
            elif(loss_type == "diffusion"):
                loss = diffusion_loss(self, y_hat, run_id)
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")
            if(self._gradients_ops(run_id, loss_type, loss, step)):
                self.nan_incidents[run_id][-1]['inputs'] = (x, y)
                loss = torch.tensor(0)
            batch_losses[run_id][loss_type] = loss
        losses = torch.stack([batch_losses[run_id][l] for l in self.tracked_losses])
        if('l0' not in self.L0[run_id]):
            self.L0[run_id]['l0'] = torch.clamp(losses.detach().clone(), max=1000)
        if VAR_IND is not None and VAR_IND not in self.L0[run_id]:
            self.L0[run_id][VAR_IND] = batch_losses[run_id]['distribution']
        return(losses)

    def _run_batch(self, x, y, phase, step):
        if self.log:
            t0 = tru.log_event("run-batch start", step=step)
        batch_losses = {}
        for run_id in self.run_ids:
            print("\n", run_id)
            batch_losses[run_id] = {}
            y_hat = self.models[run_id](x) 
            losses = self.apply_loss(y_hat, x, y, batch_losses, run_id, step)
            print("losses:", losses)
            loss_weights = self.loss_weights[run_id].detach().clone()
            print("loss_weights:", loss_weights)
            total_loss = loss_weights @ losses
            #if(step % self.training_configs[run_id].loss_weight_params['update_interval'] == 0):
            if phase == 'train':
                self.optimizers[run_id].zero_grad()
                total_loss.backward(retain_graph=True)
                if(self.training_configs[run_id].clip_gradients):
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.models[run_id].parameters(), max_norm=1.0)
                self.optimizers[run_id].step()
                if(self.lr_schedulers[run_id] is not None):
                    self.lr_schedulers[run_id].step()
                    self._update_gradnorm_weights(run_id, batch_losses[run_id])
            batch_losses[run_id]['total'] = total_loss.item()
        if(self.log):
            tru.log_event("run-batch end", duration=time.time() - t0, step=step, loss=[batch_losses[run_id]['total'] for run_id in self.run_ids])
        return(batch_losses)