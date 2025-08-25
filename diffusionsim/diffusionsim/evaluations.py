import os
import inspect
import dataclasses
import json
import numpy as np
import torch
import xarray as xr
import typing
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
from matplotlib import pyplot as plt
from ipywidgets import interact, widgets

from pprint import pprint
import copy
import diffusionsim.training_utils as tru
import diffusionsim.trainers as trainers
from collections import defaultdict

import diffusers

exp_dir = "/mnt/home/ssa2206/Climsim/experiments"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Run:
    def __init__(self, exp_id, base_run_id, climsim_run= True, cid: str = 'best-', exp_dir=exp_dir):
        self.base_dir = os.path.join(exp_dir, exp_id)
        self.base_run_id = base_run_id
        self.log_file = os.path.join(self.base_dir, f"{base_run_id}.json")
        with open(self.log_file, 'r') as f:
            self.logs = json.load(f)
        self.climsim_run = climsim_run
        self.run_ids = [key for key in self.logs.keys() if key.startswith(base_run_id)]
        #print(self.run_ids, self.logs.keys())
        self.num_runs = len(self.run_ids)
        if 'data_config' in self.logs:
            self.dconfig = self.pass_params(tru.DataConfig, self.logs['data_config'])
        else:
            self.dconfig = self.pass_params(tru.DataConfig, self.logs[self.run_ids[0]]['data_config'])
        self.mconfigs = {}
        self.tconfigs = {}
        self.models = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for run_id in self.run_ids:
            log = self.logs[run_id]
            self.mconfigs[run_id] = self.pass_params(tru.ModelConfig, log['model_config'])
            self.tconfigs[run_id] = self.pass_params(tru.TrainingConfig, log['training_config'])
            brid = self.base_run_id[:-1] if self.base_run_id[-1].isdigit() else self.base_run_id
            ckpt_dir = os.path.join(self.base_dir, "checkpoints", brid, f"{cid}{run_id}-ckpt.pt")
            if not os.path.exists(ckpt_dir):
                ckpt_dir = os.path.join(self.base_dir, "checkpoints", f"{cid}{run_id}-ckpt.pt")
            if (os.path.exists(ckpt_dir)):
                model = tru.load_model_from_ckpt(ckpt_dir, self.mconfigs[run_id], baseline=climsim_run).to(device)
                self.models[run_id] = tru.ModelLens(model)
            else:
                print(f"Checkpoint {ckpt_dir} was deleted")
                self.models[run_id] = None
        
        self.dconfig.dataloader_params.num_workers = 0
        self.dconfig.dataloader_params.prefetch_factor = None
        self.dconfig.dataloader_params.multiprocessing_context = None
        self.dconfig.dataloader_params.persistent_workers = False
    
    def write_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f)
    
    def pass_params(self, data_class, cfg_dict):
        # Start with defaults from dataclass
        valid_keys = inspect.signature(data_class).parameters
        filtered_dict = {}
        self.dict_params = {}
        for f in dataclasses.fields(data_class):
            if f.type == typing.Dict and f.name not in cfg_dict and f.default_factory is not dataclasses.MISSING:
                filtered_dict[f.name] = f.default_factory()
                self.dict_params[f.name] = f.default_factory()

        for k, v in cfg_dict.items():
            if k in valid_keys: # Direct match to a dataclass field
                filtered_dict[k] = v
            else:
                found = False
                for dp in self.dict_params:
                    if k in filtered_dict[dp]:
                        found = True
                        filtered_dict[dp][k] = v
                        break
                if not found:
                    print(f"Warning: Unknown config key '{k}' in {data_class.__name__}, ignoring.")
        return data_class(**filtered_dict)
    
    def _get_config(self, config_dict, getter):
        if getter == list or getter == 'all':
            return(list(config_dict.values()))
        elif(isinstance(getter, str) and getter in config_dict):
            return(config_dict[getter])
        elif(isinstance(getter, int)):
            return(config_dict[self.run_ids[getter]])
        else:
            raise ValueError(f"Invalid {getter}")
    
    def get_mconfig(self, getter):
        return(self._get_config(self.mconfigs, getter))
    
    def get_tconfig(self, getter):
        return(self._get_config(self.tconfigs, getter))

    def get_model(self, getter):
        return(self._get_config(self.models, getter))

    def get(self, attr, run_id=None):
        if isinstance(run_id, int):
            run_id = self.run_ids[run_id]
        configs = None
        if attr in inspect.signature(tru.TrainingConfig).parameters:
            configs = self.tconfigs
        elif attr in inspect.signature(tru.ModelConfig).parameters:
            configs = self.mconfigs
        elif attr in inspect.signature(tru.UNetParams).parameters:
            return([u[attr] for u in self.get("unet")])
        if(configs is not None):
            if run_id is not None:
                return(dataclasses.asdict(configs[run_id])[attr])
            return([dataclasses.asdict(configs[run_id])[attr] for run_id in self.run_ids])
        
        if attr in inspect.signature(tru.DataConfig).parameters:
            return(dataclasses.asdict(self.dconfig)[attr])
        elif attr in self.logs:
            return(self.logs[attr])
        elif attr in self.logs[self.run_ids[0]]:
            assert run_id is not None, f"run_id is required when attr is {attr}"
            return(self.logs[run_id][attr])
        for dp in self.dict_params:
            if attr in self.dict_params[dp]:
                configs = self.tconfigs if dp in inspect.signature(tru.TrainingConfig).parameters else self.mconfigs
                if run_id is not None:
                    return(dataclasses.asdict(configs[run_id])[dp][attr])
                return([dataclasses.asdict(configs[run_id])[dp][attr] for run_id in self.run_ids])
        print(f"Invalid {attr}")
        return None

    def print(self, attr):
        res = self.get(attr)
        for i in res:
            print(i)

    def reconstruct_trainer(self, apply_checkpoints=False, apply_indices=True, increment_run_id=True):
        if self.base_run_id[-1].isdigit():
            new_run_id = self.base_run_id[:-1] + str(int(self.base_run_id[-1]) + 1) if increment_run_id else self.base_run_id
        else:
            new_run_id = self.base_run_id + "2" if increment_run_id else self.base_run_id
        tconfigs = []
        for t in self.get_tconfig("all"):
            t_copy = copy.deepcopy(t)
            t_copy.run_id = t.run_id.replace(self.base_run_id, new_run_id)
            tconfigs.append(t_copy)
        indices = self.get("indices") if apply_indices else None
        dataloaders, indices = tru.load_dataloaders(self.dconfig, log=True, indices=indices)
        if self.climsim_run:
            trainer = trainers.ClimsimTrainer(
                dataloaders, indices, self.get_mconfig("all"), tconfigs, self.base_dir, new_run_id
            )
        else:
            trainer = trainers.DiffusionTrainer(
                dataloaders, indices, self.get_mconfig("all"), tconfigs, self.base_dir, new_run_id
            )
        
        for tid, rid in zip(trainer.run_ids, self.run_ids):
            trainer.losses[tid] = self.logs[rid]['losses']
            if apply_checkpoints and self.models[rid]:
                trainer.models[tid] = self.models[rid]

        return(trainer)


def interactive_plot(data_dict, plot_data, save_html=False):
    fig = go.Figure()
    for key in data_dict:
        for trace in data_dict[key]:
            fig.add_trace(trace)
    
    log_toggle = [
        dict(label='Linear Scale', method='relayout', args=[{'yaxis.type': 'linear'}]),
        dict(label='Log Scale', method='relayout', args=[{'yaxis.type': 'log'}]),
    ]
    fig.update_layout(
        title=plot_data['title'],
        xaxis_title=plot_data['xaxis_title'],
        yaxis_title=plot_data['yaxis_title'],
        updatemenus=[
            dict(type='buttons', direction='right', 
            showactive=True, buttons=log_toggle, x=0.0, y=1.1)
        ]
    )
    fig.update_traces(visible=True)
    if save_html:
        print(f"plotly.html")
        fig.write_html(f"/mnt/home/ssa2206/Climsim/diffusion-climsim/plotly.html")
    else:
        fig.show()


def plot_loss(run, loss_type: str='', phase: str = 'train', title: str = "", save_html=False):
    data = {}
    for run_id in run.run_ids:
        log = run.logs[run_id]
        y = log['losses'][phase][loss_type] if run.climsim_run else log['losses'][phase]
        num_batches = len(y)
        batches_per_epoch = num_batches // run.get('num_epochs', run_id)
        x = np.arange(num_batches) / max(1, batches_per_epoch)
        data[run_id] = [go.Scatter(x=x, y=y, mode='lines', name=run_id, visible=True, showlegend=True)]
    
    plot_data = dict(
        title=f"{loss_type} {phase.capitalize()} Loss, {title}",
        xaxis_title="Epochs",
        yaxis_title=f"{loss_type} Loss",
    )
    interactive_plot(data, plot_data, save_html)


def plot_gradients(run, loss_type: str, param_name: str = '0.weight', log: bool = True, save_html: bool = False):
    data = {}
    for i, run_id in enumerate(run.run_ids):
        grads = run.get("gradients", run_id)[loss_type].get(param_name, None)
        if grads is None:
            print(f"param_name {param_name} not found in gradients for run {run_id}.")
            return
        grads = np.array(grads)
        num_batches = grads.shape[0]
        batches_per_epoch = num_batches // run.get('num_epochs', run_id)
        x = np.arange(num_batches) / max(1, batches_per_epoch)
        means, stds = grads[:, 0], grads[:, 1]
        c = plotly.colors.DEFAULT_PLOTLY_COLORS[i % len(plotly.colors.DEFAULT_PLOTLY_COLORS)]

        data[run_id] = [
            go.Scatter(x=x, y=means + stds, mode='lines', line=dict(width=0, color=c), showlegend=False, hoverinfo='skip', legendgroup=run_id),
            go.Scatter(x=x, y=means - stds, fill='tonexty', line=dict(width=0, color=c), showlegend=False, hoverinfo='skip', legendgroup=run_id),
            go.Scatter(x=x, y=means, mode='lines', name=run_id, line=dict(width=2, color=c), showlegend=True, legendgroup=run_id)
        ]
    plot_data = dict(
        title=f"{loss_type} Gradient throughout training",
        xaxis_title="Epochs",
        yaxis_title="Gradient Magnitude",
    )
    interactive_plot(data, plot_data, save_html)

def hist(data, title, **kwargs):
    if(isinstance(data, torch.Tensor)):
        data = data.detach().cpu().numpy()
    plt.hist(data, **kwargs)
    if('label' in kwargs):
        plt.legend()
    plt.title(title)


def parse_log(fname):
    with open(fname) as f:
        lines = f.readlines()
    messages = []
    for line in lines:
        try:
            messages.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass
    return messages

def plot_wait_time(messages, ax, title="Time waiting"):
    if title:
        ax.set_title(title)

    wait_times = []
    end = None
    for m in messages:
        if m["event"] == "training end":
            end = m["time"]
        if m["event"] == "training start" and end is not None:
            wait_times.append(m["time"] - end)

    wait_times = np.array(wait_times)
    max_show = wait_times.mean() + 3 * wait_times.std()

    print("average wait time", wait_times.mean())

    ax.hist(wait_times, bins=np.linspace(0, max_show, 100), color="#6D0EDB")[-1]
    ax.set_xlabel("time (sec)")


def plot_log(messages, ax, title=""):
    origin = messages[0]["time"]
    # Define the rows for each event type you want to visualize
    rows = {"setup": 4, "get-item": 3, "run-batch": 2, "train": 1, "epoch": 0}
    ax.set_yticks(list(rows.values()), labels=list(rows))
    if title:
        ax.set_title(title)
    data = {"batches": [], "getitem": []}
    for m in messages:
        t = m["time"] - origin
        if m["event"] == "setup end":
            ax.barh(
                rows["setup"], m["duration"], left=t - m["duration"], edgecolor="k", linewidth=0.1, color="#6D0EDB", zorder=1,
            )
        elif m["event"] == "get-item end":
            ax.barh(
                rows["get-item"], m["duration"], left=t - m["duration"], color="#F9C846", zorder=1,
            )
            data["getitem"].append(m["duration"])
        elif m["event"] == "run-batch end":
            ax.barh(
                rows["run-batch"], m["duration"], left=t - m["duration"], color="#C396F9", zorder=1,
            )
            data["batches"].append(m["duration"])
        elif m["event"] == "training end":
            ax.barh(
                rows["train"], m["duration"], left=t - m["duration"], color="#FF6554", zorder=1,
            )
        elif m["event"] == "epoch end":
            ax.barh(
                rows["epoch"], m["duration"], left=t - m["duration"], edgecolor="k", linewidth=0.1, color="#FF9E0D", zorder=1,
            )

    ax.grid(axis="x", zorder=0, alpha=0.5)
    ax.set_xlabel("time (sec)")

    print("average batch duration", np.mean(data["batches"]))
    print("average get-item duration", np.mean(data["getitem"]))


def plot_run(fname):
    messages = parse_log(fname)
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6), width_ratios=[3, 1], dpi=400)

    plot_log(messages, axes[0], title="")
    plot_wait_time(messages, axes[1], title="")

    for m in messages:
        if m["event"] == "run start":
            text_str = "\n".join([f"{k}: {v}" for k, v in m["locals"].items() if v is not None])
            props = dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.5)
            fig.text(
                0.5,
                -0.03,
                text_str,
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="top",
                bbox=props,
            )
            break


def eval_density(x, GMM=None, training=False, return_components=False):
    if(torch.is_tensor(x)):
        x = x.detach().cpu().reshape(-1, 1)
    elif(len(x.shape) == 1):
        x = x[:, None]
    if(GMM is None):
        GMM = GaussianMixture(n_components=2)
        GMM.fit(x)
    mu = torch.tensor(GMM.means_.flatten(), dtype=torch.float64, device=x.device, requires_grad=training)[None,:] # (n_components,) (nc, d=1 flattened)
    pi = torch.tensor(GMM.weights_.flatten(), dtype=torch.float64, device=x.device, requires_grad=training)[None,:] # (n_components,)
    var = torch.tensor(GMM.covariances_.flatten(), dtype=torch.float64, device=x.device, requires_grad=training)[None,:] 
    log_probs = -0.5 * torch.log(2 * torch.pi * var) - (x - mu)**2 / (2 * var) 
    probs = torch.exp(log_probs)  # (N, K)
    weighted_probs = pi * probs
    q_x = weighted_probs.sum(dim=1, keepdim=True)  # (N, 1)
    if(return_components):
        return(q_x, weighted_probs)
    return(q_x,)

def plot_gmm_distribution(data, gmm, n_points=1000, components=False):
    if(isinstance(data, torch.Tensor)):
        data = data.detach().cpu().numpy()
    # Create a range of x values to evaluate the GMM on
    x = torch.linspace(data.min(), data.max(), n_points)
    ret = eval_density(x, gmm, return_components=components)
    # Plot histogram of actual data
    plt.hist(data, bins=50, density=True, alpha=0.5, label='Data')
    # Plot GMM density
    plt.plot(x, ret[0], 'r-', lw=2, label='GMM')
    if(components):
        plt.plot(x, ret[1][:,0], 'b-', lw=2, label='Component 1')
        plt.plot(x, ret[1][:,1], 'g-', lw=2, label='Component 2')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    return(ret[0])


def denoising_history(history, dataset, cmap='plasma'):
    # Define interactive plot function
    def plot_image(index, variable, denoise_step):
        vmap = tru.recreate_sample(history[denoise_step], dataset).isel(time=index, mlo=variable, ncol=dataset.permute_indices).data
        plt.imshow(vmap.reshape(16,24), cmap=cmap)
        plt.colorbar()
        plt.title(f"Image {index + 1} - Variable {dataset.mlo[variable].item()}")
        plt.axis("off")
        plt.show()

    # Create interactive widgets
    n, d = history[0].shape[:2]
    index_slider = widgets.IntSlider(value=0, min=0, max=n-1, step=1, description='Index:')
    variable_slider = widgets.IntSlider(value=0, min=0, max=d-1, step=1, description='Variable:')
    steps = sorted(history.keys())
    noise_slider = widgets.SelectionSlider(options=steps, value=steps[-1], description='Time Step:', continuous_update=False,)
    # Display interactive plot
    interact(plot_image, index=index_slider, variable=variable_slider, denoise_step=noise_slider)

class SampleImages(torch.utils.data.IterableDataset):
    def __init__(self, ddpm, scheduler, steps=100, batch_size=1, eta=1.0):
        self.ddpm = ddpm.to(device)
        self.scheduler = scheduler
        self.set_num_images(batch_size)
        self.eta = 1.0
        # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
        # A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) make
        # generation deterministic.
        self.num_inference_steps = steps
        
    def set_num_images(self, batch_size):
        self.batch_size = batch_size
        self.image_shape = (batch_size, self.ddpm.config.in_channels, *self.ddpm.config.sample_size)

    def __iter__(self):
        # generate a batch of images
        pipeline = diffusers.DDIMPipeline(self.ddpm, self.scheduler)
        self.scheduler.set_timesteps(self.num_inference_steps)
        while(True): # can generate infinite batches      
            image = torch.randn(self.image_shape ).to(device) # could add generator=generator
            for t in pipeline.progress_bar(self.scheduler.timesteps):
                image = self._denoise(image, t)
            yield image
    
    def _denoise(self, xt, t):
        eps_theta = self.ddpm(xt, t).sample
        x_prev = self.scheduler.step(eps_theta, t, xt, eta=self.eta)
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        return(x_prev.prev_sample, prev_timestep)
        
    def encode(self, sample, t, eps=None):
        if eps is None:
            eps = torch.randn(sample.shape, device=device) # BS x C x H x W
        xt = self.scheduler.add_noise(sample, eps, torch.LongTensor([t])) # noisy image
        return(xt, eps)
    
    def decode(self, xt, T, stride=1, track_history=True):
        stride = abs(stride)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps//stride)
        if track_history:
            history={}
        pipeline = diffusers.DDIMPipeline(self.ddpm, self.scheduler)
        for i in pipeline.progress_bar(range(T, -1, -stride)): # pipe.scheduler.timesteps[-T//stride-1:]
            if track_history:
                history[T] = xt.detach().cpu()
            xt, T = self._denoise(xt, T)
        if track_history:
            return(history)
        return(xt)

    def encode_decode(self, x0, T, stride=1, timestep_funcs={}):
        func_results = defaultdict(list)
        stride = abs(stride)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps//stride)
        pipeline = diffusers.DDIMPipeline(self.ddpm, self.scheduler)
        for t in range(0, T, stride):
            func_results["T"].append(t)
            xt, eps = self.encode(x0, t)
            func_results["xt"].append(xt.detach().cpu().clone())
            for func_name, func in timestep_funcs.items():
                func_results[func_name].append(func(xt, x0))
        while(t > 0):
            xt, t = self._denoise(xt, t)
            func_results["T"].append(t)
            func_results["xt"].append(xt.detach().cpu().clone())
            for func_name, func in timestep_funcs.items():
                func_results[func_name].append(func(xt, x0))
        return(func_results)


dist = torch.nn.MSELoss()

def mean_dist(xt, x0):
    #mu = torch.tensor(Y_mean_img, dtype=xt.dtype, device=xt.device)
    return(dist(xt, torch.zeros_like(xt)).item())

def x0_deviation_dist(xt, x0):
    return(dist(xt, x0).item()) 


timestep_funcs = {
    "mean_dist": mean_dist,
    "x0_deviation_dist": x0_deviation_dist
}