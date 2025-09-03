import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import dataclasses
import inspect
import numpy as np

def pass_config(func, data_class):
    if dataclasses.is_dataclass(data_class):
        data_class = dataclasses.asdict(data_class)
    # only pass model config params that function takes in
    accepted_params = inspect.signature(func).parameters
    filtered_kwargs = {k: v for k, v in data_class.items() if k in accepted_params}
    return func(**filtered_kwargs)

def get_activation(act_fn: str) -> nn.Module:
    ACT2CLS = {
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")

def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor.shape)`: {len(tensor.shape)} has to be 2, 3 or 4.")

def weight_init(shape: tuple, mode: str, fan_in: int, fan_out: int):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

    Returns
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    """
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, 
        init_mode: str = "kaiming_normal", init_weight: int = 1, init_bias: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(dtype=x.dtype, device=x.device).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(dtype=x.dtype, device=x.device))
        return x

class ModelLens:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.params = dict(model.named_parameters())
        self.param_names = [p for p in self.params]
    
    def get_param(self, getter):
        if isinstance(getter, str):
            return self.params[getter]
        elif isinstance(getter, int):
            return self.params[self.param_names[getter]]
        raise ValueError(f"Invalid getter: {getter}")
    
    def log_gradients(self, gdict):
        for n, p in self.params.items():
            if p.grad is not None and torch.isfinite(p.grad).all():
                gdict.setdefault(n, []).append((p.grad.mean().item(), p.grad.std().item()))
            else:
                return( dict(param_name=n, grad=p.grad, state_dict={k: v.clone().cpu() for k, v in self.params.items()}) )
        return(0)
    
    def __getattr__(self, name):
        return getattr(self.model, name)
    def __call__(self, *args):
        return self.model(*args)

    def __repr__(self):
        return repr(self.model)
    def __str__(self):
        return str(self.model)
    

def build_baseline_model(config, **kwargs):
    if(config.bl_load_model_name and "unet" in config.bl_load_model_name):
        pass
        
    elif(config.bl_load_model_name):
        mpath = os.path.join(config.bl_model_dir, config.bl_load_model_name)
        model = torch.jit.load(mpath).original_model
        return(model)
    
    layers = []
    in_dim = config.bl_input_size
    for out_dim in config.bl_hidden_dims:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, config.bl_output_size))  # Final output layer (no activation)
    model = nn.Sequential(*layers)
    return model



