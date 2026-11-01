import numpy as np
import Constants
import pandas as pd
from tqdm import tqdm

def wellside_data_label_maker(data, size, step, wells, labels, type):
    wellside_data = []
    wellside_label = np.empty([len(labels), 0, Constants.data_long])
    well_sites = pd.read_excel(Constants.well_filepath + 'Wells_location.xlsx', header=0)
    label_datas = [np.empty((0,)) for _ in range(3)]

    for well in wells:

        first = 1
        well_row = well_sites[well_sites['WELL'] == well]
        xline = well_row['xline'].values[0] - 300
        inline = well_row['inline'].values[0] - 100
        b = 0

        for label in labels:
            label_data = pd.read_csv(Constants.well_filepath + well + '.csv', sep = ',', skiprows = 0, header = 0)
            label_data = label_data[['Time', label]].values
            label_datas[b] = np.concatenate((label_datas[b], label_data[:, 1]))
            b = b + 1

            long_label = []
            for l in range(0, int(len(label_data) - Constants.data_long + 1), step):
                cube_label = label_data[l:l+Constants.data_long, 1]
                long_label.append(cube_label)

            if first == 1:
                single_wellside_label = []
                time_top = label_data[0, 0]
                time_bottom = label_data[-1, 0]
                time_top = (time_top-Constants.start) / 4
                time_bottom = (time_bottom-Constants.start) / 4 + 1
                first = 0

            single_wellside_label.append(np.array(long_label))

        wellside_label = np.concatenate((wellside_label, np.array(single_wellside_label)), axis=1)

        for time in range(int(time_top), int(time_bottom - Constants.data_long + 1), step):
            cube_data = data[int(inline - ((size - 1) / 2)):int(inline + ((size - 1) / 2) + 1), :, :][:, time:Constants.data_long + time, :][:, :, int(xline - ((size - 1) / 2)):int(xline + ((size - 1) / 2) + 1)]
            wellside_data.append(cube_data)

    if type == 'train':
        label_mean = np.zeros([len(labels)])
        label_std = np.zeros([len(labels)])
        label_datas = np.vstack(label_datas)

        for b in range(len(labels)):
            label_mean[b] = label_datas[b, :].mean()
            label_std[b] = label_datas[b, :].std()
            wellside_label[b, :, :] = (wellside_label[b, :, :] - label_mean[b]) / label_std[b]

        np.savetxt(Constants.label_mean_save, label_mean)
        np.savetxt(Constants.label_std_save, label_std)

    elif type == 'test':
        label_mean = np.loadtxt(Constants.label_mean_save)
        label_std = np.loadtxt(Constants.label_std_save)

        for b in range(len(labels)):
            wellside_label[b, :, :] = (wellside_label[b, :, :] - label_mean[b]) / label_std[b]

    wellside_data = np.array(wellside_data)
    wellside_label = np.transpose(wellside_label, [1, 2, 0])

    return wellside_data, wellside_label

def wellside_only_data_maker(data, size, step, xlines, inlines, bottom, labels):

    wellside_data = []

    for i in tqdm(range(len(xlines))):

        xline = xlines[i]
        inline = inlines[i]

        for depth in range(0, int(bottom - Constants.data_long + 1), step):
            cube_data = data[int(inline-((size-1)/2)):int(inline+((size-1)/2)+1),:,:][:,depth:Constants.data_long+depth,:][:,:,int(xline-((size-1)/2)):int(xline+((size-1)/2)+1)]
            wellside_data.append(cube_data)

    wellside_data = np.array(wellside_data)

    wellside_label = np.full((len(wellside_data), Constants.data_long, len(labels)), -999)

    return wellside_data, wellside_label

def random_xlines_inlines(num, inline_lim, xline_lim):
    inlines = np.random.randint(inline_lim[0], inline_lim[1], num)
    xlines = np.random.randint(xline_lim[0], xline_lim[1], num)

    return xlines, inlines