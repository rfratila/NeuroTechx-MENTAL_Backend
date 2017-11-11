import numpy as np
import scipy.io as sio
from vulcanai.net import Network
from vulcanai.utils import *
import theano.tensor as T
import os

from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from random import shuffle

import json

#TODO:
#have a fully convolutional layer to account for variable recording time

#inputMat = sio.loadmat('%scovertShiftsOfAttention_VPiac.mat'%data_path)# got this from: http://bnci-horizon-2020.eu/database/data-sets
# def getData(data_path):

#     training_set =[]

#     for patient in os.listdir(data_path):
#         if patient.endswith('.mat'):
#             training_set.append(patient)

#     print ("%i samples found"%len(training_set))

#     trainOut = [[1,0],[0,1]]*len(training_set) #this will contain the actual state of the brain

#     data =[]
#     for patient in training_set:
#         temp = sio.loadmat('%s%s'%(data_path,patient))
#         data.append(temp['data']['X'][0][0][:1000])

#     data = np.stack(data)
#     trainOut = np.stack(trainOut)
#     data = OrderedDict(input=np.array(data, dtype='float32'), truth=np.array(trainOut, dtype='float32'))
#     return data

#be able to read from an Attentive folder and create their truth values
def get_json_data(data_path):
    if 'inattentive' in data_path:
        train_out = np.array([0, 1]) #this will contain the actual state of the brain: inattentive
    else:
        train_out = np.array([1, 0]) #this will contain the actual state of the brain: attentive
    data = []
    res = {}
    with open(data_path) as infile:
        res = json.load(infile)
    for timeStamp in res['data']:
        data.append(np.array(timeStamp['channel_values'], dtype='float32'))
    data = np.stack(data, axis=1)
    data = np.resize(data, (data.shape[0], 300000))
    data = OrderedDict(input=np.array(data, dtype='float32'),
                       truth=np.array(train_out, dtype='float32'))
    return data


def get_all_json_data(path_list):
    data = defaultdict(list)
    for sample in path_list:
        temp = get_json_data(sample)
        for key, value in temp.iteritems():
            data[key].append(value)
    return data


def show_eeg(sample, saliency_map=None, title="Behind the scenes"):
    if saliency_map is not None:
        if sample.shape != saliency_map.shape:
            raise ValueError('Sample and saliency map not same shape.')
    channels = sample.shape[0]
    last_datapoint = int(0.3 * sample.shape[1])
    plt.figure()
    fig, axes = plt.subplots(nrows=channels, ncols=1)
    for i, ax in enumerate(axes):
        plt.subplot(channels, 1, i + 1)
        plt.plot(sample[i][:last_datapoint], alpha=0.7)
        if saliency_map is not None:
            split_ratio = 0.01
            _samples_per_split = int(split_ratio * last_datapoint)
            _splits = [j * _samples_per_split for j in range(int(1 / split_ratio))]
            min_range = int(min(sample[i][:last_datapoint]))
            s_map = saliency_map[i][:last_datapoint]
            split_s_map = np.array([np.average(s_map[c * _samples_per_split: (c + 1) * _samples_per_split]) for c in range(int(1 / split_ratio))])
            colours = cm.hot_r(split_s_map / float(max(split_s_map)))
            plt.bar(
                _splits,
                [min_range] * len(_splits),
                [_samples_per_split] * len(_splits),
                alpha=0.7,
                color=colours,
                edgecolor='none',
                linewidth=0,
            )

    c_axis = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
    mpl.colorbar.ColorbarBase(c_axis, cmap='hot_r', norm=norm, orientation='vertical')
    plt.suptitle(title)
    plt.show(False)

#input: person_Name, timeIntervalBetweenSamples
#output: json file with date,personName (YYYY/MM/DD/HH/DD)

def main():
    data_path = 'data'

    test_reserve = 0.0
    validation_reserve = 0.2
    training_reserve = 1 - (test_reserve + validation_reserve)
    input_var = T.tensor3('input')
    y = T.dmatrix('truth')

    data_set = []

    for patient in [data_path]:
        attentive_path = os.path.join(data_path, 'attentive')
        inattentive_path = os.path.join(data_path, 'inattentive')
        if os.path.exists(attentive_path) and os.path.exists(inattentive_path):
            data_set += \
                [os.path.join(attentive_path, i) for i in os.listdir(attentive_path)]
            data_set += \
                [os.path.join(inattentive_path, i) for i in os.listdir(inattentive_path)  if i.endswith('.json')]
            shuffle(data_set)

    print ("%i samples found" % len(data_set))

    training_set = data_set[:int(training_reserve * len(data_set))]
    test_set = data_set[int(training_reserve * len(data_set)):-int(test_reserve * len(data_set))]
    validation_set = data_set[int(test_reserve * len(data_set) + int(training_reserve * len(data_set))):]

    example = get_json_data(training_set[0])

    network_conv_config = {
        'mode': 'conv',
        'filters': [16],
        'filter_size': [[5]],
        'stride': [[1]],
        'pool': {
            'mode': 'average_exc_pad',
            'stride': [[2]]
        }
    }

    conv_net = Network(
        name='eeg_conv',
        dimensions=[None] + list(example['input'].shape),
        input_var=input_var,
        y=y,
        config=network_conv_config,
        input_network=None,
        num_classes=2,
        learning_rate=0.0001)

    train_eeg = get_all_json_data(training_set)
    validation_eeg = get_all_json_data(validation_set)

    # x = np.reshape(np.array(train_eeg['input']),
    #                [np.array(train_eeg['input']).shape[0], 8 * 300000])
    # display_tsne(x, get_class(np.array(train_eeg['truth'])).flatten())

    conv_net.train(
        epochs=2,
        train_x=np.array(train_eeg['input']),
        train_y=np.array(train_eeg['truth']),
        val_x=np.array(validation_eeg['input']),
        val_y=np.array(validation_eeg['truth']),
        batch_ratio=0.2,
        plot=True
    )

    example = 1
    pred = conv_net.forward_pass(
        np.array(validation_eeg['input'])[example: example + 1])
    pred_str = "Attentive" if pred[0, 0] == 1 else "Inattentive"
    truth_str = "Attentive" if validation_eeg['truth'][0][0] == 1 else "Inattentive"
    title = "Prediction: {} | Truth: {}".format(pred_str, truth_str)

    m = get_saliency_map(conv_net, np.array(validation_eeg['input']))
    show_eeg(np.array(validation_eeg['input'])[example], m[example], title=title)

if __name__ == "__main__":
    main()
