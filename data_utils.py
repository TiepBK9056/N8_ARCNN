import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import torch.nn.functional as F

from utils import (
    load_wav_to_torch,
    load_filepaths,
    wav_to_3d_mel,
)

class AudioLabelLoader(torch.utils.data.Dataset):
    def __init__(self, hparams, set_name):
        self.audiopaths_and_label = load_filepaths(
                                        os.path.join(hparams.common.meta_file_folder, "{}.csv".format(set_name))
                                    )
        self.sampling_rate = hparams.data.sampling_rate
        self.num_filter_bank = hparams.data.num_filter_bank
        self.list_classes = hparams.data.classes
        self.num_classes = len(self.list_classes)
        self.max_length = hparams.data.max_length
        
        random.seed(1234)
        random.shuffle(self.audiopaths_and_label)

    def get_audio(self, filename):
        waveform, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        data = wav_to_3d_mel(
                    wav=waveform,
                    sampling_rate=sampling_rate,
                    num_filter=self.num_filter_bank,
                    max_length=self.max_length
                )
        return data
        
    def get_audio_label_pair(self, audiopath_label):
        audiopath, label = audiopath_label[0], audiopath_label[1]
        data = self.get_audio(audiopath)
        label_one_hot = F.one_hot(torch.tensor(self.list_classes.index(label)), self.num_classes)
        return (data, label_one_hot)

    def __getitem__(self, index):
        return self.get_audio_label_pair(self.audiopaths_and_label[index])

    def __len__(self):
        return len(self.audiopaths_and_label)