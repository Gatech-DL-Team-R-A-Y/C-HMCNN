import torch
import torch.nn as nn
from .helper import get_constr_out


class AW_CNN(nn.Module):
    def __init__(self, R):
        super(AW_CNN, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        # Take the idea and adaptation of VGG16

        self.R = R
        self.number_conv_block = 3

        initial_in_channel = 1
        initial_out_channel = 16
        ks = 3  # kernal size
        fc_hidden_size = 2048

        in_c, out_c = initial_in_channel, initial_out_channel
        self.conv_blocks = nn.ModuleList([])
        for _ in range(self.number_conv_block):
            self.conv_blocks.append(
                nn.ModuleList([
                    nn.Conv2d(in_c, out_c, ks, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(out_c, out_c, ks, 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ])
            )
            in_c, out_c = out_c, 2 * out_c

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        linear_size = 3 * in_c * 2 ** (2 * (8 - self.number_conv_block))
        # print('AW_debug linear_size', linear_size)

        self.fc1 = nn.Linear(linear_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, 399)
        self.sigmoid = nn.Sigmoid()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        # x = x.unsqueeze(0)
        for i in range(self.number_conv_block):
            for j in range(5):
                x = self.conv_blocks[i][j](x)
                # print('Model AW Debug (i, j):', i, j, x.shape)
            # x = self.conv_blocks[i][0](x)
            # x = self.conv_blocks[i][1](x)
            # x = torch.relu(x)
            # x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
        # if self.training:
        #     constrained_out = x
        # else:
        #     constrained_out = get_constr_out(x, self.R)
        # return constrained_out