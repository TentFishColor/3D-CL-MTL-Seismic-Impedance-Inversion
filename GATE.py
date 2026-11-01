import torch
import torch.nn as nn
import Constants

class GATE(nn.Module):
    def __init__(self, input_size, output_size):
        super(GATE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Conv3d(in_channels = input_size, out_channels = output_size,
                              kernel_size = (Constants.size, Constants.size, Constants.size), stride = (1, 1, 1), padding = (int((Constants.size-1)/2), int((Constants.size-1)/2), int((Constants.size-1)/2)), bias = False)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        x = self.conv(input)
        x = self.softmax(x)
        x = x.transpose(1, 2).transpose(2, 3).transpose(3, 4)

        return x


def GATE_model(input_channels, output_channels):
    model = GATE(input_size = input_channels, output_size = output_channels).cuda()

    return model
