import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidalDownsample(nn.Module):
    """
    Pyramid downsampling layer that concatenates adjacent time frames.
    Reduces temporal resolution by 2x while doubling feature dimension.
    """
    def __init__(self):
        super(PyramidalDownsample, self).__init__()
   
    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
       
        if seq_len % 2 == 1:
            x = x[:, :-1, :]
            seq_len = seq_len - 1
       
        x = x.contiguous().view(batch_size, seq_len // 2, 2, feat_dim)
        x = x.contiguous().view(batch_size, seq_len // 2, 2 * feat_dim)
       
        return x

class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer that computes weighted average of all time steps.
    """
    def __init__(self, input_dim, attention_units=128):
        super(AttentionPooling, self).__init__()
        self.attention_units = attention_units
        self.attention_dense = nn.Linear(input_dim, attention_units)
        self.attention_score = nn.Linear(attention_units, 1)
   
    def forward(self, x, mask=None):
        attention_hidden = torch.tanh(self.attention_dense(x))
        attention_scores = self.attention_score(attention_hidden)
       
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, -1e9)
       
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_output = torch.sum(attention_weights * x, dim=1)
       
        return attended_output
