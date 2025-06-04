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