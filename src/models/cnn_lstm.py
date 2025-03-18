import torch.nn as nn

from Project.src.models.cnn import CNN
from Project.src.models.lstm import LSTM


class CNN_LSTM(nn.Module):
    def __init__(
            self,
            cnn_input_size,
            cnn_output_size,
            cnn_channels,
            cnn_kernel_size,
            lstm_layer_size,
            lstm_layers,
            final_output_size,
            dropout=0.3,
            ):

        super().__init__()

        self.cnn = CNN(input_size=cnn_input_size,
                       output_size=cnn_output_size,
                       out_channels_1=cnn_channels,
                       kernel_size=cnn_kernel_size)

        self.lstm = LSTM(input_size=1,
                         hidden_size=lstm_layer_size,
                         num_layers=lstm_layers,
                         output_size=final_output_size,
                         dropout=dropout)

    def forward(self, x):

        out = self.cnn(x)
        out = out.unsqueeze(-1)
        out = self.lstm(out)

        return out
