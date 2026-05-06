import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from .components import PyramidalDownsample, AttentionPooling
from ..config import device

class PyramidalBiLSTM(nn.Module):
    """
    Pyramidal BiLSTM model for audio deepfake detection with attention pooling.
    """
    def __init__(self, input_dim, base_units=128, num_pyramid_layers=2,
                 dropout_rate=0.3, recurrent_dropout=0.2, attention_units=128, dense_units=64):
        super(PyramidalBiLSTM, self).__init__()
       
        self.input_dim = input_dim
        self.base_units = base_units
        self.num_pyramid_layers = num_pyramid_layers
       
        self.first_bilstm = nn.LSTM(
            input_dim, base_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.first_dropout = nn.Dropout(dropout_rate)
        self.first_layernorm = nn.LayerNorm(base_units * 2)
       
        self.pyramid_layers = nn.ModuleList()
        self.pyramid_bilstms = nn.ModuleList()
        self.pyramid_dropouts = nn.ModuleList()
        self.pyramid_layernorms = nn.ModuleList()
       
        current_dim = base_units * 2
        for i in range(num_pyramid_layers):
            self.pyramid_layers.append(PyramidalDownsample())
            current_dim = current_dim * 2
           
            bilstm = nn.LSTM(
                current_dim, base_units,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0
            )
            self.pyramid_bilstms.append(bilstm)
           
            dropout_rate_progressive = min(dropout_rate + 0.05 * (i + 1), 0.6)
            self.pyramid_dropouts.append(nn.Dropout(dropout_rate_progressive))
            self.pyramid_layernorms.append(nn.LayerNorm(base_units * 2))
           
            current_dim = base_units * 2
       
        self.attention_pooling = AttentionPooling(base_units * 2, attention_units)
        self.attention_dropout = nn.Dropout(dropout_rate + 0.1)
       
        self.dense1 = nn.Linear(base_units * 2, dense_units)
        self.dense1_layernorm = nn.LayerNorm(dense_units)
        self.dense1_dropout = nn.Dropout(min(dropout_rate + 0.2, 0.7))
       
        self.output = nn.Linear(dense_units, 1)
   
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
       
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None
       
        if lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.first_bilstm(packed_input)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            x, _ = self.first_bilstm(x)
       
        x = self.first_layernorm(x)
        x = self.first_dropout(x)
       
        current_mask = mask
       
        for i in range(self.num_pyramid_layers):
            x = self.pyramid_layers[i](x)
           
            if current_mask is not None:
                if current_mask.size(1) % 2 == 1:
                    current_mask = current_mask[:, :-1]
                current_mask = current_mask[:, ::2]
           
            x, _ = self.pyramid_bilstms[i](x)
            x = self.pyramid_layernorms[i](x)
            x = self.pyramid_dropouts[i](x)
       
        x = self.attention_pooling(x, current_mask)
        x = self.attention_dropout(x)
       
        x = F.relu(self.dense1(x))
        x = self.dense1_layernorm(x)
        x = self.dense1_dropout(x)
       
        x = self.output(x)
       
        return x

def build_pyramidal_bilstm(input_shape, base_units=128, num_pyramid_layers=2,
                          dropout_rate=0.3, recurrent_dropout=0.2, attention_units=128,
                          dense_units=64, learning_rate=1e-4, weight_decay=1e-5):
    """
    Build and return PyTorch Pyramidal BiLSTM model with optimizer
    """
    input_dim = input_shape[1] if isinstance(input_shape, tuple) else input_shape
   
    model = PyramidalBiLSTM(
        input_dim=input_dim,
        base_units=base_units,
        num_pyramid_layers=num_pyramid_layers,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        attention_units=attention_units,
        dense_units=dense_units
    ).to(device)
   
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-7
    )
   
    criterion = nn.BCEWithLogitsLoss()
   
    return model, optimizer, criterion
