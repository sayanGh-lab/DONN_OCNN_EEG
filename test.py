# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:39:47 2024

@author: sinha
"""

import numpy as np
from matplotlib import pyplot as plt
import mne
import os
print(__doc__)
import scipy.io
import cv2
data5=scipy.io.loadmat(r'C:/Users/sinha/Desktop/epileptic_OCNN/sub5/model_dataset.mat')

EEG_data_trn=data5["EEG_main"] # 4 channels EEG TRAINING Data

label_trn=data5["label"]

idx=data5["index"]

print(EEG_data_trn.shape,"raw_data trn", idx.shape,"index")
data = np.zeros((64, 1))
output_root_dir = 'C:/Users/sinha/Desktop/epileptic_OCNN/sub5/Data1'
for i in range(414):
  print(i)
  e1=EEG_data_trn[i,:]

  print(e1.shape)

  output_dir = os.path.join(output_root_dir, f'trial_{i}')
  os.makedirs(output_dir, exist_ok=True)


  for ii in range(64):
    eeg2=e1[ii]
    filename = os.path.join(output_dir, f'topomap_sample_{ii}.png')
    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    n_channels = len(biosemi_montage.ch_names)
    fake_info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250.,
                            ch_types='eeg')
    data[6] =  eeg2[0]  # f7
    data[12] = eeg2[1]  # c3
    data[20] = eeg2[2]  # p3
    data[39] = eeg2[0]  # f4
    data[49] = eeg2[4]  # c4
    data[57] = eeg2[5]  # p4
    data[41] = eeg2[6]  # fp2
    data[47] = eeg2[7]  # cz
    data[30] = eeg2[8]  # pz
    data[59] = eeg2[9]  # p8
    data[5] = eeg2[10]  # f8
    data[51] = eeg2[11]  # t8
    data[63] = eeg2[12]  # o2
    data[6] = eeg2[13]  # f7
    data[14] = eeg2[14]  # t7
    data[7] = eeg2[15]  # ft9/ft7
    data[42] = eeg2[16]  # ft10/ft8
    cnorm = {'kind': 'value', 'vmin': 0.2, 'vmax': 0.8}
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(biosemi_montage)
    chs = ['Oz', 'Fpz', 'T7', 'T8']
    montage_head = fake_evoked.get_montage()
    ch_pos = montage_head.get_positions()['ch_pos']
    pos = np.stack([ch_pos[ch] for ch in chs])
    radius = np.abs(pos[[2, 3], 0]).mean()
    x = pos[0, 0]
    y = pos[-1, 1]
    z = pos[:, -1].mean()
    fig, ax = plt.subplots(ncols=1, figsize=(2, 1), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)
    plt.rcParams["savefig.pad_inches"] = 0
    plt.rcParams["savefig.bbox"] = 'tight'
    o = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax,
                         show=False,sensors=False, outlines=None, res=32,sphere=None,
                         border=0, vlim=(-0.2, 0.8),  contours=0,)
    fig.savefig(filename)
    plt.close(fig)
