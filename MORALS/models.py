import torch
from torch import nn
import math
from torch.nn import functional as F
# TODO: Separate parameters for each model?

class Encoder(nn.Module):
    def __init__(self, config):
        num_layers = config['num_layers'] if 'num_layers' in config else 2
        hidden_shape = config['hidden_shape'] if 'hidden_shape' in config else 32
        if 'high_dims' in config:
            input_shape = config['high_dims']
        else:
            raise ValueError("high_dims not specified in config")
        if 'low_dims' in config:
            lower_shape = config['low_dims']
        else:
            raise ValueError("low_dims not specified in config")
        
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.encoder.add_module(f"linear_{i}", nn.Linear(input_shape, hidden_shape))
            else:
                self.encoder.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            self.encoder.add_module(f"relu_{i}", nn.ReLU(True))
        self.encoder.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, lower_shape))
        self.encoder.add_module(f"tanh_{num_layers}", nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        num_layers = config['num_layers'] if 'num_layers' in config else 2
        hidden_shape = config['hidden_shape'] if 'hidden_shape' in config else 32
        if 'high_dims' in config:
            input_shape = config['high_dims']
        else:
            raise ValueError("high_dims not specified in config")
        if 'low_dims' in config:
            lower_shape = config['low_dims']
        else:
            raise ValueError("low_dims not specified in config")

        super(Decoder, self).__init__()

        self.decoder = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.decoder.add_module(f"linear_{i}", nn.Linear(lower_shape, hidden_shape))
            else:
                self.decoder.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            self.decoder.add_module(f"relu_{i}", nn.ReLU(True))
        self.decoder.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, input_shape))
        self.decoder.add_module(f"sigmoid_{num_layers}", nn.Sigmoid())

    def forward(self, x):
        x = self.decoder(x)
        return x

class LatentDynamics(nn.Module):
    # Takes as input an encoding and returns a latent dynamics
    # vector which is just another encoding
    def __init__(self, config):
        num_layers = config['num_layers'] if 'num_layers' in config else 2
        hidden_shape = config['hidden_shape'] if 'hidden_shape' in config else 32
        if 'low_dims' in config:
            lower_shape = config['low_dims']
        else:
            raise ValueError("low_dims not specified in config")

        super(LatentDynamics, self).__init__()

        self.dynamics = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.dynamics.add_module(f"linear_{i}", nn.Linear(lower_shape, hidden_shape))
            else:
                self.dynamics.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            self.dynamics.add_module(f"relu_{i}", nn.ReLU(True))
        self.dynamics.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, lower_shape))
        self.dynamics.add_module(f"tanh_{num_layers}", nn.Tanh())
    
    def forward(self, x):
        x = self.dynamics(x)
        return x


class MaskedTransformer(nn.Module):
    def __init__(self, config):
        super(MaskedTransformer, self).__init__()

        input_size = config['input_size']
        embed_size = config['embed_size']
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        max_sequence_length = config['max_sequence_length']
        num_layers = config['num_layers']
        
        self.batch_first = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        # embed_size = 2, low_dims

        # self.linear_in = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh(), nn.Linear(128, 32),  nn.Tanh(), nn.Linear(32, embed_size))  # removed embed_size // 2 for no cnn
        self.linear_in = nn.Sequential(nn.Linear(input_size, 8), nn.Linear(8, embed_size))  # removed embed_size // 2 for no cnn

        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                        dim_feedforward=hidden_size, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Classification head
        # self.fc_classification = nn.Linear(latent_dim, output_dim)

        # self.linear_out = nn.Sequential(nn.Linear(embed_size, 32), nn.Tanh(), nn.Linear(32, 128), nn.Tanh(), nn.Linear(128, input_size))
        self.linear_out = nn.Sequential(nn.Linear(embed_size, 8), nn.Linear(8, input_size))

    def forward_transformer_enc(self, lin_input):
        x = self.get_latent_embeddings(lin_input) # encoder
        x = self.positonal_embedding(x)
        x = self.encoder(x) # dynamics
        return x

    def forward(self, lin_input):
        x = self.forward_transformer_enc(lin_input)
        # x = self.activation(x)
        x = self.linear_out(x) # decoder
        return x

    def get_latent_embeddings(self, lin_input):
        return self.linear_in(lin_input) # encoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)


class NewMaskedTransformer(nn.Module):
    def __init__(self, config):
        super(NewMaskedTransformer, self).__init__()

        input_size = config['input_size']
        embed_size = config['embed_size']
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        max_sequence_length = config['max_sequence_length']
        num_layers = config['num_layers']
        
        self.batch_first = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        # embed_size = 2, low_dims

        # self.linear_in = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh(), nn.Linear(128, 32),  nn.Tanh(), nn.Linear(32, embed_size))  # removed embed_size // 2 for no cnn
        self.positonal_embedding = PositionalEncoding(input_size, max_len=max_sequence_length)

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                             dim_feedforward=hidden_size, batch_first=self.batch_first),
                                            num_layers) # in 4 and gives out 4
        
        
        self.linear_in = nn.Sequential(nn.Linear(input_size, embed_size), nn.Tanh())

        self.temporal_fc = nn.Sequential(nn.Linear(embed_size, embed_size), nn.Tanh())

        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads),
            num_layers=num_layers
        )

        self.linear_out =nn.Sequential(nn.Linear(embed_size, input_size)) # reconstruction
        self.tanh_activation = nn.Tanh()
        
        # self.linear_out = nn.Sequential(nn.Linear(embed_size, 8), nn.Linear(8, input_size))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        

    def forward(self, lin_input):
        pos_x = self.positonal_embedding(lin_input) # 4-dim
        enc_x = self.encoder(pos_x) # mu, log_var
        z_t = self.linear_in(enc_x)
        # print(z_t.shape, lin_input.shape) 
        dec_z = self.decoder(z_t.unsqueeze(1).repeat(1, lin_input.size(1), 1))
        x_pred = self.linear_out(dec_z)
        return x_pred

    def get_latent_embeddings(self, lin_input):

        x = self.positonal_embedding(lin_input) # 4-dim
        # x = self.get_latent_embeddings(lin_input) # encoder
        x = self.encoder(x) # mu, log_var
        # x = self.tanh_activation(x)
        z_t = self.linear_in(x)

        return z_t

