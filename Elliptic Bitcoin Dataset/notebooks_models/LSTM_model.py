import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        batch_size = input.size(0)
        lstm_out, _ = self.lstm(input.view(batch_size, -1, self.input_dim))  # view function is used to reshape the tensor
        # lstm_out holds the output of the last layer for all timesteps, with shape [batch_size, seq_len, hidden_dim]
        # we take the output from the last timestep with lstm_out[:, -1, :]
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred
