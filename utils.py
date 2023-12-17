import numpy as np
import torch
import torchaudio
import wave
import random
import yaml
import os
import glob

import python_speech_features as ps

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # Get name of last checkpoint
    checkpoint_folder = os.path.dirname(checkpoint_path)
    last_checkpoint = glob.glob(os.path.join(checkpoint_folder, '*.pt'))
    # Save best checkpoint
    torch.save({'model': state_dict,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                }, checkpoint_path)
    # Remove last checkpoint
    if len(last_checkpoint) != 0:
        os.remove(last_checkpoint[0])


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict['model'])
    print("Continue from epoch: {}".format(epoch))
    return model, optimizer, epoch


def load_wav_to_torch(full_path):
    waveform, sampling_rate = torchaudio.load(full_path, channels_first=False, normalize=False)
    return waveform, sampling_rate


def normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data


def wav_to_3d_mel(wav, sampling_rate, num_filter, max_length=300):
    # Get Log-MelSpectrogram, delta, and delta-delta feature
    mel_spec = ps.logfbank(wav, sampling_rate, nfilt = num_filter) 
    detal1 = ps.delta(mel_spec, 2)
    detal2 = ps.delta(detal1, 2)

    data = np.empty((max_length, num_filter, 3))
    time = mel_spec.shape[0]
    # Normalize data with z-score
    new_mel_spec = normalization(mel_spec)
    new_detal1 = normalization(detal1)
    new_detal2 = normalization(detal2)
    if time <= max_length:
        # Pad MelSpectrogram which is shorter than max_length
        new_mel_spec = np.pad(
            array=new_mel_spec,
            pad_width=((0, max_length - new_mel_spec.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        new_detal1 = np.pad(
            array=new_detal1,
            pad_width=((0, max_length - new_detal1.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        new_detal2 = np.pad(
            array=new_detal2,
            pad_width=((0, max_length - new_detal2.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        data[:,:,0] = new_mel_spec
        data[:,:,1] = new_detal1
        data[:,:,2] = new_detal2
    else:
        # Get random start frame
        start_frame = random.randint(0, time - max_length)
        end_frame = start_frame + max_length
        data[:,:,0] = new_mel_spec[start_frame:end_frame,:]
        data[:,:,1] = new_detal1[start_frame:end_frame,:]
        data[:,:,2] = new_detal2[start_frame:end_frame,:]
    
    return torch.from_numpy(data)


def load_filepaths(filename, split=','):
    with open(filename, encoding='utf-8') as f:
        next(f)     # Remove header line
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load(stream, Loader=yaml.FullLoader)
    return docs


class Dotdict(dict):
    
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='config.yaml'):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__