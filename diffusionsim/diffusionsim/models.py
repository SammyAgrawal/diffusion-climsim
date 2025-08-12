import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field
from . import DIFFUSERS_AVAILABLE
import os
import json
from typing import Optional, Tuple, Union
import inspect
from . import blocks_1d

if DIFFUSERS_AVAILABLE:
    from . import diffusers
    from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps

def move_device(model, new_device):
    print("Moving model to ", new_device)
    model.device = new_device
    model = model.to(new_device)
    return(model)


def load_model(config, **kwargs):
    registered = ['VAE', 'diffusion', 'latent_diffusion']
    def pass_config(func, data_class):
        # only pass model config params that function takes in
        accepted_params = inspect.signature(func).parameters
        filtered_kwargs = {k: v for k, v in asdict(data_class).items() if k in accepted_params}
        return func(**filtered_kwargs)
        
    match config.model_type.lower():
        case "vae":
            model = VariationalAutoencoder(
                data_dims= config.num_channels, 
                latent_dims= config.latent_dims, 
                hidden_dims= config.ae_hidden_dims,
                disable_logstd_bias = config.disable_enc_logstd_bias,
            )
        case model_type if "diffusion" in model_type:
            if 'latent' in model_type: # modify channels for VAE 
                config.unet.in_channels = config.latent_dims 
                config.unet.out_channels = config.latent_dims
            model = pass_config(diffusers.UNet2DModel, config.unet)
        case "baseline":
            model = build_baseline_model(config)
        case _:
            raise ValueError(f"Model type {config.model_type} not supported")
    if('device' in kwargs):
        model = model.to(kwargs["device"])
    if("distributed" in kwargs and kwargs["distributed"]):
        assert "rank" in kwargs, "rank must be provided if distributed is True"
        model = nn.parallel.DistributedDataParallel(model, device_ids=[kwargs["rank"]])
    
    return(model)

def build_baseline_model(config, **kwargs):
    if(config.bl_load_model_name):
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

class TestCNN(torch.nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class VariationalEncoder(torch.nn.Module):
    """
    Conditional VAE Encoder with <layers>+1 fully connected layer
    """
    def __init__(self, in_dims, hidden_dims=[64, 32, 16], latent_dims=16, logstd_bias=False, dropout=0, device='cpu'):
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(in_dims, hidden_dims[0])])
        for j in range(len(hidden_dims)-1):
            self.linears.append(nn.Linear(hidden_dims[j], hidden_dims[j+1]))
                
            #self.linears += [torch.nn.Sequential(
            #    torch.nn.Linear(in_dims if i == 0 else hidden_dims, hidden_dims),
            #    torch.nn.LayerNorm(hidden_dims),
            #    torch.nn.Dropout(p=dropout))
            #    ]
        self.enc_mean = torch.nn.Linear(hidden_dims[-1], latent_dims)
        self.enc_logstd = torch.nn.Linear(hidden_dims[-1], latent_dims, bias=logstd_bias)
        self.kl = 0
        self.device = device

    def forward(self, x):
        #if (type(x) != torch.Tensor):
        #    x = torch.tensor(x, dtype=torch.float32, )
        #z = torch.flatten(x.squeeze(), start_dim=1)
        z = x.squeeze()
        for linear in self.linears:
            z = F.relu(linear(z))
        mu, sigma = self.enc_mean(z), torch.exp(self.enc_logstd(z)) # ensures sigma is always positive
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean() # mean so that kl div comparable to mse
        return(mu, sigma)

class Decoder(torch.nn.Module):
    """
    Conditional VAE Decoder with <layers>+1 fully connected layer
    """
    def __init__(self, out_dims, hidden_dims=[16, 32, 64], latent_dims=16):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(latent_dims, hidden_dims[0])])
        for j in range(len(hidden_dims)-1):
            self.linears.append(nn.Linear(hidden_dims[j], hidden_dims[j+1]))
        self.dec_mu = nn.Linear(hidden_dims[-1], out_dims)
        #self.dec_std = torch.nn.Linear(hidden_dims[-1], out_dims)

    def forward(self, z):
        for linear in self.linears:
            z = torch.nn.functional.relu(linear(z))
        mu = self.dec_mu(z)
        #sig = torch.exp(self.dec_std(z))
        return mu

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, data_dims=128, label_dims=128,
                 latent_dims=16, hidden_dims=[64, 32, 16], disable_logstd_bias=False):
        """
        Conditional VAE
        Encoder: [y x] -> [mu/sigma] -sample-> [z]
        Decoder: [z x] -> [y_hat]

        Inputs:
        -------
        beta - [float] trade-off between KL divergence (latent space structure) and reconstruction loss
        data_dims - [int] size of x
        label_dims - [int] size of y
        latent_dims - [int] size of z
        hidden_dims - [int] size of hidden layers
        layers - [int] number of layers, including hidden layer
        """
        super().__init__()
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        self.encoder = VariationalEncoder(data_dims, hidden_dims, latent_dims, logstd_bias= not disable_logstd_bias)
        decoder_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(data_dims, decoder_hidden, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.to(device)
        #self.N.scale = self.N.scale.to(device)

    def forward(self, x, sample=True):
        # Normalize
        #if batch_norm:
        #    x_m, x_s = x.mean(axis=0), x.std(axis=0)
        #    mx = x_s != 0
        #    x[:, mx] = (x[:, mx] - x_m[mx]) / x_s[mx] 
            
        mu, sigma = self.encoder(x)
        if(sample):
            z = mu + sigma * torch.randn(sigma.shape, device=self.device)
        else:
            z = mu
        x_hat = self.decoder(z)
            #if batch_norm:
            #    y_hat_mean = (y_hat_mean + y_m) * y_s
        return x_hat

    def sample(self, x, random=True):
        """
        Sample conditionally on x

        Inputs:
        -------
        x - [BxN array] label
        random - [boolean] if true sample latent variable from prior else use all-zero vector
        """
        if random:
            # Draw from prior
            z = self.encoder.N.sample([x.shape[0], self.latent_dims])
        else:
            # Set to prior mean
            z = torch.zeros([x.shape[0], self.latent_dims]).to(device)
        mean_y, std_y = self.decoder(z, x)
        if random:
            # add output noise
            y = mean_y + self.N.sample(mean_y.shape) * std_y
            # y = torch.zeros_like(mean_y)
            # nz = torch.rand(y.shape).to(device) > p0
            # y[nz] = mean_y[nz] + self.encoder.N.sample([(nz == 1).sum()]) * std_y[nz]
            return y
        else:
            return mean_y, std_y


@dataclass
class UNet1DOutput(diffusers.utils.BaseOutput):
    sample: torch.Tensor



# https://github.com/leap-stc/ClimSim/blob/main/online_testing/baseline_models/Unet_v5/training/climsim_unet.py


class ClimsimUNet1DModel(diffusers.models.ModelMixin, diffusers.configuration_utils.ConfigMixin):
    r"""
    A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0):
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model was initially designed for.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`):
            Tuple of block output channels.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): Optional output processing block of UNet.
        act_fn (`str`, *optional*, defaults to `None`): Optional activation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        downsample_each_block (`int`, *optional*, defaults to `False`):
            Experimental feature for using a UNet without upsampling.
    """

    _skip_layerwise_casting_patterns = ["norm"]

    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        sample_rate: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "fourier",
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = False,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        act_fn: str = None,
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
    ):
        super().__init__()
        self.sample_size = sample_size

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        if use_timestep_embedding:
            time_embed_dim = block_out_channels[0] * 4
            self.time_mlp = TimestepEmbedding(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=block_out_channels[0],
            )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1

            down_block = blocks_1d.get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = blocks_1d.get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=block_out_channels[0],
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1] if i < len(up_block_types) - 1 else final_upsample_channels
            )

            is_final_block = i == len(block_out_channels) - 1

            up_block = blocks_1d.get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = blocks_1d.get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
        debug=False,
    ) -> Union[UNet1DOutput, Tuple]:
        r"""
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_1d.UNet1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_1d.UNet1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed.to(sample.dtype))
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(sample.dtype)
            timestep_embed = timestep_embed.broadcast_to((sample.shape[:1] + timestep_embed.shape[1:]))
        if debug:
            print(sample.shape, timestep_embed.shape)
        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=timestep_embed)
            down_block_res_samples += res_samples

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=timestep_embed)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)
