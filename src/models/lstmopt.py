import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer


class CVXLayer(nn.Module):
    def __init__(self,
                 problem,
                 parameters: list,
                 variables: list):
        super().__init__()

        self.cvxpylayer = CvxpyLayer(problem,
                                     parameters=parameters,
                                     variables=variables)

    def forward(self, *x):
        y = self.cvxpylayer(*x)

        return y


class LSTMOPT(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
            dropout,
            problem,
            parameters,
            variables,
            scaler,
            ):
        """
        Simple OLD_LSTM model made in pytorch
        :param input_size: the size of the input (based on the lags provided)
        :param hidden_size: the hidden layer sizes
        :param num_layers: the number of layers in the OLD_LSTM (each of size hidden_size)
        :param output_size: the forecast window (f.e. 24 means 'forecast 24 hours')
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.cvxlayer = CVXLayer(problem, parameters, variables)
        self.scaler = scaler
        self.relu = nn.ReLU()

    def forward(self, input, *parameters):
        """
        Forward method for the OLD_LSTM layer. I.e. how input gets processed
        :param input: the input tensor
        :return: output tensor
        """
        hidden, _ = self.lstm(input, None)
        if hidden.dim() == 2:
            hidden = hidden[-1, :]
        else:
            hidden = hidden[:, -1, :]
        output = self.linear(hidden)
        output = self.relu(output)
        rescaled_output = output * (self.scaler[1] - self.scaler[0]) + self.scaler[0]

        opt_output = self.cvxlayer(rescaled_output, *parameters)

        return output, opt_output
