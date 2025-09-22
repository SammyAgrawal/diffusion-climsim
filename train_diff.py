import os
import sys
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
import time
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim.trainers import DiffusionTrainer
import diffusionsim.models as models
import torch
import pprint

def setup_diffusion_run(
    num_models, exp_id, base_run_id, 
    data_vars, 
    image_dim,
    batch_size=128, 
    shuffle_indices=False, 
    use_tendencies=False, source='local-vzarr'
):
    dconfig = tru.my_dconfig(
        source=source,
        data_vars=data_vars,
        in_notebook=in_notebook,
        shuffle_indices=shuffle_indices,
        dataset_type=dataset_type,
        batch_size=batch_size,
        use_tendencies=use_tendencies
    )
    dconfig.phases = ['train']
    dconfig.train_test_split = [0.40]
    tconfigs, mconfigs = [], []
    dconfig.batch_checkpoint_interval = 32
    dconfig.batch_logging_interval = 128
    learning_rates = [1e-4] * num_models

    unet_types = ["nv"] * num_models
    num_blocks = [3, 3, 3, 3, 3, 3]
    kernel_sizes = [3, 3, 3, 5, 5, 5]
    block_out_channels = [
        (64, 128, 128, 128),
        (64, 128, 256, 256),
        (64, 256, 512, 512),
        #(64, 128, 128, 128),
        #(64, 128, 256, 256),
        #(64, 256, 512, 512)
    ]

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=f"{base_run_id}{lettering[i]}", optimizer='adam-groups')
        tconfig.param_groups = [
            dict(select_method='exclude', keyword='scale_output'),
            dict(select_method='include', keyword='scale_output', lr=1e-1, weight_decay=0.0),
        ]
        tconfig.phases = ['train']
        tconfig.learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE
        tconfig.lr_scheduler = "cosine"
        tconfig.lr_warmup_steps = 75
        tconfig.max_T_sample = 100
        tconfig.clip_gradients = True
        tconfigs.append(tconfig)

        mconfig = tru.my_unet(
            block_channels = block_out_channels[i], 
            utype=unet_types[i],
            data_vars=data_vars,
            image_dim=image_dim,
            kernel_size=kernel_sizes[i],
            num_blocks=num_blocks[i]
        )
        mconfigs.append(mconfig)
    return(tconfigs, mconfigs, dconfig)

trainer = None
num_epochs = 5
REF_BATCH_SIZE = 64
dataset_type = "climsim_diffusion1d"
image_dim = 1
data_vars = 'v1'
in_notebook = True
NUM_MODELS = 3

RESTART_FROM_CKPT = False
RERUN = False
torch.set_grad_enabled(True)

EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
if __name__ == "__main__":
    exp_id = "diffusion_hp_search"
    run_id = "v1_modulus"
    run_start_time = tru.log_event("run start", exp_id=exp_id, run_id=run_id)
    t0 =  tru.log_event("setup start", run_id=run_id)
    if(RERUN):
        run = diff.evaluations.Run(exp_id, run_id, climsim_run=False, cid='')
        run.dconfig.dataloader_params.batch_size = 256
        trainer = run.reconstruct_trainer(apply_checkpoints=True, apply_indices=True)
    else:
        base_dir = os.path.join(EXP_DIR, exp_id)
        tconfigs, mconfigs, dconfig = setup_diffusion_run(
            NUM_MODELS, exp_id, run_id, data_vars, image_dim, batch_size=256, shuffle_indices=False, use_tendencies=False
        )
        dataloaders, indices = tru.load_dataloaders(dconfig, log=True)
        scheduler = models.SchedulerParams(
            model_type = "ddpm-scheduler",
            beta_schedule = "linear",
            clip_sample = False,
            clip_sample_range = 4.0,
            beta_end = 0.02,
            prediction_type = "v_prediction"
        )
        scheduler = models.load_model(scheduler, apply_lens=False)
        trainer = DiffusionTrainer(dataloaders, indices, mconfigs, tconfigs, base_dir, run_id, scheduler)
    
    #ckpt = "/mnt/home/ssa2206/Climsim/experiments/diffusion_hp_search/checkpoints/diff_1d/diff_1da-ckpt.pt"
    #state = torch.load(ckpt, weights_only=True, map_location=trainer.device)
    #trainer.models[trainer.run_ids[0]].load_state_dict(state)

    tru.log_event("setup end", duration=time.time() - t0)
    if (RESTART_FROM_CKPT):
        trainer.restart_from_ckpt(num_epochs, log=True, cid='')
    else:
        trainer.train(num_epochs, log=True)
    print("Done!", flush=True)
    tru.log_event("run end", duration = time.time() - run_start_time)
