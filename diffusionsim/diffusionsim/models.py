import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field
try:
    import diffusers
except:
    print("Diffusers not in workspace, not all models can be loaded")

def move_device(model, new_device):
    print("Moving model to ", new_device)
    model.device = new_device
    model = model.to(new_device)
    return(model)

import inspect
def load_model(config):
    registered = ['VAE', 'diffusion', 'latent_diffusion']
    def pass_config(func, data_class):
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
            return(model)
        case model_type if "diffusion" in model_type:
            if 'latent' in model_type:
                config.unet.in_channels = config.latent_dims 
                config.unet.out_channels = config.latent_dims
            return(pass_config(diffusers.UNet2DModel, config.unet))
        
    return(-1)


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





