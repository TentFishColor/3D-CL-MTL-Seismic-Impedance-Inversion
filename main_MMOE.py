import torch
import torch.nn as nn
import numpy as np
import os
import math
import random
import time
import Constants
from torch.utils.data.dataset import Dataset
from traindata_and_testdata_maker import wellside_data_label_maker, wellside_only_data_maker, random_xlines_inlines
from segy import readsegy
from MMOE import MMOE_model
from UNET import UNET_Forward
from loss_calculation import loss_calculation_pyramid, loss_calculation_wavelet, loss_calculation_impedance, loss_calculation_forward, loss_calculation_PCC
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def seed_torch(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    h = math.floor(m / 60)
    s -= m * 60
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


class dataset_maker(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        return data, label, index

    def __len__(self):
        return len(self.label)


class dataset_maker_onlydata(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __getitem__(self, index):
        data = self.data[index]

        return data, index

    def __len__(self):
        return len(self.data)


def train(trainloader):
    all_losses = []
    all_vail_losses = []
    all_accuracy = []
    start = time.time()
    current_loss = 0
    current_accuracy = 0
    train_mode = Constants.train_mode

    for iter in range(Constants.n_iters):

        if train_mode == 'backward':
            net_backward.train()
            net_forward.eval()
        elif train_mode == 'forward':
            net_backward.eval()
            net_forward.train()
        elif train_mode == 'back&for':
            net_backward.train()
            net_forward.train()

        train_loss = 0
        accuracy = 0

        for i, (data, label, indices) in enumerate(trainloader):
            data, label = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(label.cuda())

            if train_mode == 'backward':
                optimizer_backward.zero_grad()
            elif train_mode == 'forward':
                optimizer_forward.zero_grad()
            elif train_mode == 'back&for':
                optimizer_backward.zero_grad()
                optimizer_forward.zero_grad()

            output1, output2, output3 = net_backward.forward(data)
            outputs = torch.cat((output1, output2, output3), dim = 1)
            output_forward = net_forward.forward(outputs)

            loss1 = loss_calculation_pyramid(output=output1, label=label[:, :, 0].unsqueeze(-1), size=Constants.size,
                                             criterion=criterion)
            loss2 = loss_calculation_pyramid(output=output2, label=label[:, :, 1].unsqueeze(-1), size=Constants.size,
                                             criterion=criterion)
            loss3 = loss_calculation_pyramid(output=output3, label=label[:, :, 2].unsqueeze(-1), size=Constants.size,
                                             criterion=criterion)

            loss_pyramid = loss1 + loss2 + loss3
            loss_impedance = loss_calculation_impedance(V=output1, DEN=output2, Z=output3)
            loss_forward = loss_calculation_forward(sim_data=output_forward, data=data, criterion=criterion)

            if Constants.step == 3:
                loss = loss_pyramid
            else:
                loss = loss_pyramid + loss_forward + loss_impedance

            loss.backward()

            if train_mode == 'backward':
                optimizer_backward.step()
            elif train_mode == 'forward':
                optimizer_forward.step()
            elif train_mode == 'back&for':
                optimizer_backward.step()
                optimizer_forward.step()

            train_loss += float(loss) / len(data)

            accuracy += 1-float(abs(data - output_forward).sum() / len(label) / Constants.data_long)

        train_loss = train_loss / len(trainloader)
        accuracy = accuracy / len(trainloader)

        current_loss += train_loss
        current_accuracy += accuracy

        if (iter + 1) % Constants.plot_every == 0:
            all_losses.append(current_loss / Constants.plot_every)
            all_accuracy.append(current_accuracy / Constants.plot_every)
            current_loss = 0

        if (iter + 1) % Constants.print_every == 0:
            print("Iter:", iter + 1, "Time:", timeSince(start),
                  "\nTrain loss:", train_loss,
                  "\nTrain accuracy:", accuracy)

    return all_losses, all_vail_losses


def test(testloader):
    with torch.no_grad():
        net_backward.eval()
        net_forward.eval()

        test_loss = 0
        test_accuracy1 = 0
        test_accuracy2 = 0
        test_accuracy3 = 0

        for i, (data, label, indices) in enumerate(testloader):
            data, label = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(label.cuda())
            output1, output2, output3 = net_backward.forward(data)
            outputs = torch.cat((output1, output2, output3), dim = 1)
            output_forward = net_forward.forward(outputs)

            loss1 = loss_calculation_pyramid(output=output1, label=label[:, :, 0].unsqueeze(-1), size=Constants.size,
                                             criterion=criterion)
            loss2 = loss_calculation_pyramid(output=output2, label=label[:, :, 1].unsqueeze(-1), size=Constants.size,
                                             criterion=criterion)
            loss3 = loss_calculation_pyramid(output=output3, label=label[:, :, 2].unsqueeze(-1), size=Constants.size,
                                             criterion=criterion)

            loss_pyramid = loss1 + loss2 + loss3
            loss_impedance = loss_calculation_impedance(V=output1, DEN=output2, Z=output3)
            loss_forward = loss_calculation_forward(sim_data=output_forward, data=data, criterion=criterion)
            loss = loss_pyramid + loss_impedance + loss_forward

            test_loss += float(loss) / len(label)

            test_accuracy1 += 1-float((abs(label[:, :, 0].unsqueeze(-1) - output1[:, :, int((Constants.size-1)/2), :, int((Constants.size-1)/2)].to(torch.float32).transpose(1, 2))).sum() / len(label) / Constants.data_long)
            test_accuracy2 += 1-float((abs(label[:, :, 1].unsqueeze(-1) - output2[:, :, int((Constants.size-1)/2), :, int((Constants.size-1)/2)].to(torch.float32).transpose(1, 2))).sum() / len(label) / Constants.data_long)
            test_accuracy3 += 1-float((abs(label[:, :, 2].unsqueeze(-1) - output3[:, :, int((Constants.size-1)/2), :, int((Constants.size-1)/2)].to(torch.float32).transpose(1, 2))).sum() / len(label) / Constants.data_long)

        test_loss = test_loss / len(testloader)
        test_accuracy1 = test_accuracy1 / len(testloader)
        test_accuracy2 = test_accuracy2 / len(testloader)
        test_accuracy3 = test_accuracy3 / len(testloader)

        print("Test loss", test_loss,
              "\nTest accuracy1", test_accuracy1, "||Test accuracy2", test_accuracy2, "||Test accuracy3", test_accuracy3)

    return test_loss



seed_torch(seed = Constants.seed)
first = 1
criterion = nn.MSELoss().cuda()
X_trains = []
X_tests = []

for arr in Constants.seismic_arrs:

    seismic_data = readsegy(filename_in = Constants.arr_filepath + arr + '.sgy', inline_byte = 9, xline_byte = 21)
    np.savetxt(Constants.data_mean_save, np.expand_dims(seismic_data.mean(), axis=0))
    np.savetxt(Constants.data_std_save, np.expand_dims(seismic_data.std(), axis=0))

    seismic_data = (seismic_data-seismic_data.mean())/seismic_data.std()

    X_train, Y_train = wellside_data_label_maker(data = seismic_data, size = Constants.size, step = Constants.step, wells = Constants.train_wells, labels = Constants.label_used, type = 'train')

    if Constants.unlabeled_num != 0 :
        xlines, inlines = random_xlines_inlines(num=Constants.unlabeled_num, inline_lim=[(Constants.size - 1) / 2, seismic_data.shape[0] - ((Constants.size - 1) / 2) - 50], xline_lim=[(Constants.size - 1) / 2, seismic_data.shape[2] - ((Constants.size - 1) / 2)])
        X_train2, Y_train2 = wellside_only_data_maker(data=seismic_data, size=Constants.size, step=Constants.step, xlines=xlines, inlines=inlines, bottom=seismic_data.shape[1], labels = Constants.label_used)

        X_train = np.concatenate((X_train, X_train2), axis=0)
        Y_train = np.concatenate((Y_train, Y_train2), axis=0)

    X_test, Y_test = wellside_data_label_maker(data = seismic_data, size = Constants.size, step = Constants.step, wells = Constants.test_wells, labels = Constants.label_used, type = 'test')

    X_trains.append(X_train)
    X_tests.append(X_test)

X_trains = np.transpose(np.array(X_trains), [1, 0, 2, 3, 4])
X_tests = np.transpose(np.array(X_tests), [1, 0, 2, 3, 4])
trainset = dataset_maker(data = X_trains, label = Y_train)
testset = dataset_maker(data = X_tests, label = Y_test)

del X_trains, X_tests, Y_test

trainloader = torch.utils.data.DataLoader(trainset, batch_size = Constants.train_batch_size, shuffle = True, pin_memory = True, drop_last = False)
testloader = torch.utils.data.DataLoader(testset, batch_size = Constants.test_batch_size, shuffle = False, pin_memory = True, drop_last = False)

del trainset, testset

net_forward = UNET_Forward(input_channels = len(Constants.label_used))
net_backward = MMOE_model()

optimizer_forward = torch.optim.Adam(net_forward.parameters(), lr = Constants.learning_rate, weight_decay = Constants.weight_decay)
optimizer_backward = torch.optim.Adam(net_backward.parameters(), lr = Constants.learning_rate, weight_decay = Constants.weight_decay)

# net_backward.load_state_dict(torch.load('net_B'))
# net_forward.load_state_dict(torch.load('net_F'))

all_losses, all_vail_losses = train(trainloader = trainloader)
test_loss = test(testloader = testloader)

torch.save(net_backward.state_dict(), Constants.B_net_save)
torch.save(net_forward.state_dict(), Constants.F_net_save)