import torch
import torch.nn as nn
from newmodel import ODEGCN



class MLPDiffusion(nn.Module):
    def __init__(self, num_nodes,n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(num_nodes, num_units),
                nn.Tanh(),
                nn.Linear(num_units, num_units),
                nn.Tanh(),
                nn.Linear(num_units, num_units),
                nn.Tanh(),
                nn.Linear(num_units, num_nodes),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x


class TCNDiffusion(nn.Module):
    def __init__(self, num_nodes,n_steps, sequence_length,num_units=128, num_layers=2, kernel_size=3, dropout=0.1):
        super(TCNDiffusion, self).__init__()
        
        self.input_channels = num_nodes
        self.output_channels = self.input_channels
        self.sequence_length = sequence_length
        layers = []
        in_channels = self.input_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, self.output_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = self.output_channels
        self.network = nn.Sequential(*layers)


        self.fcn = nn.ModuleList(
            [
            nn.Linear(self.output_channels * self.sequence_length, num_units),  # 压缩为 512 维
            nn.Tanh(),
            nn.Linear(num_units, num_units),
            nn.Tanh(),
            nn.Linear(num_units, num_units),
            nn.Tanh(),
            nn.Linear(num_units, self.output_channels * self.sequence_length)  # 解码为原始形状
            ]
        )
        
        # self.sequence_length = sequence_length
        # self.input_channels = input_channels

        # self.linears = nn.ModuleList(
        #     [
        #         nn.Linear(num_nodes, num_units),
        #         nn.Tanh(),
        #         nn.Linear(num_units, num_units),
        #         nn.Tanh(),
        #         nn.Linear(num_units, num_units),
        #         nn.Tanh(),
        #         nn.Linear(num_units, num_nodes),
        #     ]
        # )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )


    # def forward(self, x):
    #     return self.network(x)
    def forward(self, x, t):
        x.view(x.size(0), self.input_channels, self.sequence_length)
        x = self.network(x)
        x = x.view(x.size(0), -1)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            # x = self.network(x)
            # x = x.view(x.size(0), -1)
            x = self.fcn[2 * idx](x)
            x += t_embedding
            x = self.fcn[2 * idx + 1](x)

        x = self.fcn[-1](x)

        return x.view(x.size(0), self.input_channels, self.sequence_length)





class EMA():

    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


class DiffTCNODE(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat,n_steps, num_units=128):
        super(DiffTCNODE, self).__init__()

        self.ODEGCN = ODEGCN(num_nodes = num_nodes, num_features = num_features,
                             num_timesteps_input = num_timesteps_input, num_timesteps_output = num_timesteps_output
                             ,A_sp_hat = A_sp_hat)
        self.diffusion =  TCNDiffusion( num_nodes = num_nodes, n_steps = n_steps, sequence_length=num_timesteps_input,num_units= num_units)
        self.n_steps = n_steps

    def forward(self, x, t,choose):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)

        difx : input data of shape(batch_size,num_nodes)
        """
        difx = x[:, :, :, :1]
        difx = difx.squeeze()

        dif_pre_x = self.diffusion(difx, t)
        # dif
        if choose <1:
            x_t = self.ODEGCN(x)
            return x_t, dif_pre_x
        else:
            return dif_pre_x
        # if t < (self.n_steps/4):
        #     x_t = self.ODEGCN(x)
        #     return x_t , dif_pre_x
        # else:
        #     return  dif_pre_x

