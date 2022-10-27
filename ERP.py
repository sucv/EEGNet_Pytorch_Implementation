import os
# mne imports
import mne
from mne import io
from mne.datasets import sample


import numpy as np
import torch

import model as md
import utils as utils

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

data_path = sample.data_path()

# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for saving the model
current_working_dir = os.getcwd()
filename = current_working_dir + '/best_model.pth'
# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data() * 1000  # format is in (trials, channels, samples)
y = labels

# Configure the network particulars
kernels, chans, samples = 1, 60, 151
n_classes, dropoutRate, kernelLength, kernelLength2, F1, D = 4, 0.5, 64, 16, 8, 2
F2 = F1 * D

# take 50/25/25 percent of the data to train/validate/test
X_train = X[0:144, ]
Y_train = y[0:144]
X_validate = X[144:216, ]
Y_validate = y[144:216]
X_test = X[216:, ]
Y_test = y[216:]

############################# EEGNet portion ##################################
Y_train = torch.nn.functional.one_hot(torch.as_tensor(Y_train - 1, dtype=torch.int64)).to(device)
Y_validate = torch.nn.functional.one_hot(torch.as_tensor(Y_validate - 1, dtype=torch.int64)).to(device)
Y_test = torch.nn.functional.one_hot(torch.as_tensor(Y_test - 1, dtype=torch.int64)).to(device)

# convert data to NCHW (trials, kernels, channels, samples) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = torch.as_tensor(X_train.reshape(X_train.shape[0], kernels, chans, samples),  dtype=torch.float32).to(device)
X_validate = torch.as_tensor(X_validate.reshape(X_validate.shape[0], kernels, chans, samples), dtype=torch.float32).to(device)
X_test = torch.as_tensor(X_test.reshape(X_test.shape[0], kernels, chans, samples), dtype=torch.float32).to(device)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

net = md.EEGNet(n_classes, chans, samples, dropoutRate, kernelLength, kernelLength2, F1, D, F2)
net = net.to(device)
print(md.torch_summarize(net))


optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
batch_size = 64
# net = nn.DataParallel(net)
net.cuda()

best_acc = 0
for iter in range(1, 300):
    running_loss = 0
    running_error = 0
    num_batches = 0

    permutation = torch.randperm(X_train.size()[0])

    # create a minibatch
    net.train()
    for batch_index in range(0, X_train.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[batch_index:batch_index + batch_size]
        minibatch_data = X_train[indices]
        minibatch_label = Y_train[indices]

        # reshape them to fit the network
        inputs = minibatch_data
        # feed the input to the net
        inputs.requires_grad_()
        scores = net(inputs)

        # update the weights
        minibatch_label = minibatch_label.type(torch.cuda.FloatTensor)
        loss = md.categorical_cross_entropy(scores, minibatch_label)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()
        minibatch_label_non_hot = minibatch_label.argmax(axis=-1).cuda()
        error = utils.get_error ( scores.detach(), minibatch_label_non_hot)
        running_error += error.item()

        num_batches += 1

    total_loss = running_loss/num_batches
    total_error = running_error/num_batches

    # validation
    net.eval()
    inputs = X_validate
    val_probs = net(inputs)
    val_preds = val_probs.argmax(axis=-1).cuda()
    val_acc = np.mean((val_preds == Y_validate.argmax(axis=-1)).double().cpu().numpy())

    if val_acc > best_acc:
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename)
        # net.save_state({'state_dict':net.state_dict(), 'optimizer': optimizer.state_dict()}, os.getcwd())
        best_acc = val_acc
        print('best=%.5f' % best_acc)
    print('epoch=%d, \t loss=%.5f, \t error=%.9f, \t val_acc=%.5f.' % (int(iter+1),  total_loss, total_error, val_acc))

X_test = X_test.type(torch.FloatTensor).cuda()
checkpoint = torch.load(filename)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
net.eval()
probs = net(X_test.cuda())
preds = probs.argmax(axis = -1).cuda()
acc = np.mean((preds == Y_test.argmax(axis=-1)).double().cpu().numpy())
print("Classification accuracy: %f " % (acc))


