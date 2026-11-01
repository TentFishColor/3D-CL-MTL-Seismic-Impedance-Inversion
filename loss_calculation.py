import torch
import torch.nn.functional as F
import numpy as np
import Constants
import torch.nn as nn

def loss_calculation_pyramid(output, label, size, criterion):

    mask = label[:, 0, 0] != -999
    masked_outputs = output[mask]
    masked_labels = label[mask]

    if mask.sum() == 0:
        loss = 0
        return loss

    masked_labels = masked_labels.to(torch.float32)
    main = masked_outputs[:, :, int((size - 1) / 2), :, int((size - 1) / 2)]
    main = main.to(torch.float32).transpose(1, 2)
    main_loss = criterion(main, masked_labels)

    # weight = (0.8 / int((size - 1) / 2))
    weight = 0.2
    surround_losses = torch.empty([int((size - 1) / 2)])

    for i in range(int((size - 1) / 2)):
        surround = masked_outputs[:, :, i:size-i, :, i:size-i]
        surround = surround.to(torch.float32)
        s_labels = torch.empty(surround.shape).cuda()

        for j in range(surround.shape[2]):

            for k in range(surround.shape[4]):
                s_labels[:,:,j,:,k] = masked_labels.transpose(1, 2)

        surround_losses[i] = criterion(surround, s_labels) * weight
        weight = weight * 2

    surround_loss = torch.sum(surround_losses)

    loss = main_loss + surround_loss
    # loss = main_loss

    return loss

def loss_calculation_impedance(V, DEN, Z):

    label_mean = np.loadtxt(Constants.label_mean_save)
    label_std = np.loadtxt(Constants.label_std_save)
    criterion = nn.MSELoss().cuda()

    v = (V * label_std[0]) + label_mean[0]
    den = (DEN * label_std[1]) + label_mean[1]
    vd = v * den
    VD = (vd - label_mean[2]) / label_std[2]

    loss = criterion(VD, Z)

    return loss

def loss_calculation_forward(sim_data, data, criterion):

    loss = criterion(sim_data, data)

    return loss
