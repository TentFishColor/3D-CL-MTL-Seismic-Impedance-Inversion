import torch
import torch.nn as nn
import numpy as np
from UNET import UNET_Expert, UNET_Tower
from GATE import GATE_model
import Constants

class Expert_Gate(nn.Module):
    def __init__(self, feature_dim, expert_dim, n_expert, n_task):
        super(Expert_Gate, self).__init__()
        self.n_task = n_task

        self.expert_layers = nn.ModuleList(
            [UNET_Expert(input_channels=feature_dim, output_channels=expert_dim) for _ in range(n_expert)])
        self.gate_layers = nn.ModuleList(
            [GATE_model(input_channels=feature_dim, output_channels=n_expert) for _ in range(n_task)])


    def forward(self, x):
        E_net = [expert(x) for expert in self.expert_layers]
        E_net = torch.cat(([e[:, :, :, :, :, np.newaxis].transpose(1, 2).transpose(2, 3).transpose(3, 4) for e in E_net]), dim=-1)
        gate_net = [gate(x) for gate in self.gate_layers]

        towers = []
        for i in range(self.n_task):
            g = gate_net[i].unsqueeze(-1)
            tower = torch.matmul(E_net, g)
            tower = tower.squeeze().transpose(4, 3).transpose(3, 2).transpose(2, 1)
            towers.append(tower)

        return towers

class MMoE(nn.Module):
    def __init__(self, feature_dim, expert_dim, n_expert, n_task):
        super(MMoE, self).__init__()

        self.Expert_Gate = Expert_Gate(feature_dim=feature_dim, expert_dim=expert_dim, n_expert=n_expert, n_task=n_task)

        self.tower1 = UNET_Tower(input_channels=expert_dim, output_channels=Constants.output_channels)
        self.tower2 = UNET_Tower(input_channels=expert_dim, output_channels=Constants.output_channels)
        self.tower3 = UNET_Tower(input_channels=expert_dim, output_channels=Constants.output_channels)

    def forward(self, x):

        towers = self.Expert_Gate(x)
        out1 = self.tower1(towers[0])
        out2 = self.tower2(towers[1])
        out3 = self.tower3(towers[2])

        return out1, out2, out3

def MMOE_model():
    model = MMoE(feature_dim=Constants.input_channels, expert_dim = Constants.expert_dim, n_expert = Constants.n_expert, n_task = Constants.n_task).cuda()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of MMOE parameters: %d' % nParams)
    return model

