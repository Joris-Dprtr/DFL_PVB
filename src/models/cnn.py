from torch import nn


class CNN(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            out_channels_1,
            kernel_size
            ):

        super().__init__()

        self.conv_layer_1 = nn.Conv1d(in_channels=1, out_channels=out_channels_1, kernel_size=kernel_size)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv_layer_2 = nn.Conv1d(in_channels=out_channels_1, out_channels=out_channels_1, kernel_size=kernel_size)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        output_length_1 = (input_size - kernel_size) // 1 + 1
        output_length_pool_1 = (output_length_1 - 2) // 2 + 1

        output_length_2 = (output_length_pool_1 - kernel_size) // 1 + 1
        output_length_pool_2 = (output_length_2 - 2) // 2 + 1

        out_channels_2 = out_channels_1
        linear_input_size = out_channels_2 * output_length_pool_2

        self.fc_1 = nn.Linear(linear_input_size, out_features=output_size)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        out = self.conv_layer_1(x)
        out = self.max_pool_1(out)

        out = self.conv_layer_2(out)
        out = self.max_pool_2(out)

        out = self.fc_1(out.reshape(out.size(0), -1))

        return out
