arr_filepath = 'E:/Datasets/F3/seismic/'
well_filepath = 'E:/Datasets/F3/wells/'
save_filepath = 'E:/Datasets/F3/results/'

label_mean_save = 'E:/MTL-F3/Mean&Std/MMOE_label_mean.txt'
label_std_save = 'E:/MTL-F3/Mean&Std/MMOE_label_std.txt'
data_mean_save = 'E:/MTL-F3/Mean&Std/MMOE_data_mean.txt'
data_std_save = 'E:/MTL-F3/Mean&Std/MMOE_data_std.txt'

seismic_arrs = ['']
start = 408
label_used = ['V', 'DEN', 'Z']
train_wells = ['F02-1', 'F03-4', 'F03-2']
test_wells = ['F06-1']
unlabeled_num = 1000

train_mode = 'backward' #'backward' 'forward' 'back&for'
B_net_save = 'net_B'
F_net_save = 'net_F'
seed = 3407
n_iters = 100  
print_every = 1 
plot_every = 1 

size = 7
step = 1
data_long = 32
input_channels = len(seismic_arrs)
output_channels = 1

expert_dim = 64
n_expert = 6
n_task = len(label_used)

learning_rate = 0.0001 
weight_decay = 1e-6

train_batch_size = 128
test_batch_size = 128
