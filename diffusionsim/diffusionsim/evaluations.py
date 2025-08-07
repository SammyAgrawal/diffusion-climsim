import os
import inspect
from dataclasses import asdict
import json
import numpy as np
import torch
import xarray as xr

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


exp_dir = "/mnt/home/ssa2206/Climsim/experiments"




class Run:
    def __init__(self, exp_id, base_run_id, cid: str = 'best-'):
        
        self.base_dir = os.path.join(exp_dir, exp_id)
        self.base_run_id = base_run_id
        self.log_file = os.path.join(self.base_dir, f"{base_run_id}.json")
        with open(self.log_file, 'r') as f:
            self.logs = json.load(f)
        self.run_ids = [key for key in self.logs.keys() if key.startswith(base_run_id)]
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
                model = tru.load_model_from_ckpt(ckpt_dir, self.mconfigs[run_id], baseline=True).to(device)
                self.models[run_id] = tru.ModelLens(model)
            else:
                print(f"Checkpoint {ckpt_dir} was deleted")
                self.models[run_id] = None
    
    def write_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f)
    
    def pass_params(self, data_class, dict):
        dict = {k: v for k, v in dict.items() if k in inspect.signature(data_class).parameters}
        return(data_class(**dict))
    
    def print_dconfig(self):
        pprint(self.dconfig)
    
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
        if attr in inspect.signature(tru.TrainingConfig).parameters:
            configs = self.tconfigs
        elif attr in inspect.signature(tru.ModelConfig).parameters:
            configs = self.mconfigs
        elif attr in inspect.signature(tru.DataConfig).parameters:
            return(asdict(self.dconfig)[attr])
        elif attr in self.logs:
            return(self.logs[attr])
        elif attr in self.logs[self.run_ids[0]]:
            assert run_id is not None, f"run_id is required when attr is {attr}"
            return(self.logs[run_id][attr])
        else:
            print(f"Invalid {attr}")
            return None
        if run_id is not None:
            return(asdict(configs[run_id])[attr])
        return([asdict(configs[run_id])[attr] for run_id in self.run_ids])

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
        trainer = trainers.ClimsimTrainer(self.dconfig, self.get_mconfig("all"), tconfigs,
                                 torch.nn.MSELoss(), self.base_dir, new_run_id, indices=indices)
        
        for run_id in trainer.run_ids:
            trainer.losses[run_id] = self.logs[run_id]['losses']
        if apply_checkpoints:
            trainer.models = self.models

        return(trainer)
    


def interactive_plot(data_dict, plot_data):
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
    fig.show()


def plot_loss(diff_logs, loss_type, phase='train', title=""):
    keys = [k for k in diff_logs.keys() if k != 'indices' and k != 'data_config']
    data = {}
    for key in keys:
        log = diff_logs[key]
        y = log['losses'][phase][loss_type]
        num_batches, batches_per_epoch = len(y), len(y) // log['training_config']['num_epochs']
        #print(f"{batches_per_epoch=}, {num_batches=}, epochs={tconfig['num_epochs']=}")
        x = np.arange(num_batches) / max(1, batches_per_epoch)
        data[key] = [go.Scatter(x=x, y=y, mode='lines', name=key, visible=True, showlegend=True)]
    plot_data = dict(
        title=f"{loss_type} Training Loss, {title}",
        xaxis_title="Epochs",
        yaxis_title=f"{loss_type} Loss",
    )
    interactive_plot(data, plot_data)

def plot_loss(run, loss_type: str, phase: str = 'train', title: str = ""):
    data = {}
    for run_id in run.run_ids:
        log = run.logs[run_id]
        y = log['losses'][phase][loss_type]
        num_batches = len(y)
        batches_per_epoch = num_batches // run.get('num_epochs', run_id)
        x = np.arange(num_batches) / max(1, batches_per_epoch)
        data[run_id] = [go.Scatter(x=x, y=y, mode='lines', name=run_id, visible=True, showlegend=True)]
    
    plot_data = dict(
        title=f"{loss_type} {phase.capitalize()} Loss, {title}",
        xaxis_title="Epochs",
        yaxis_title=f"{loss_type} Loss",
    )
    interactive_plot(data, plot_data)


def plot_gradients(run, loss_type: str, param_name: str = '0.weight', log: bool = True):
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
            go.Scatter(x=x, y=means + stds, mode='lines', line=dict(width=0, color=c), showlegend=False, hoverinfo='skip', legendgroup=key),
            go.Scatter(x=x, y=means - stds, fill='tonexty', line=dict(width=0, color=c), showlegend=False, hoverinfo='skip', legendgroup=key),
            go.Scatter(x=x, y=means, mode='lines', name=key, line=dict(width=2, color=c), showlegend=True, legendgroup=key)
        ]
    plot_data = dict(
        title=f"{loss_type} Gradient throughout training",
        xaxis_title="Epochs",
        yaxis_title="Gradient Magnitude",
    )
    interactive_plot(data, plot_data)

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



