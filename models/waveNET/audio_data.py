import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import os

class WaveNetDataset(Dataset):
    def __init__(self, 
                 dataset_path,
                 out_path, 
                 input_length, 
                 target_length=1, 
                 sampling_rate=16000, 
                 mono=True):
        
        self.dataset_path = dataset_path
        self.out_path = out_path
        self.sampling_rate = sampling_rate
        self.input_length = input_length
        self.target_length = target_length
        self.mono = mono
        self.sample_loc_util = [0]
        
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"dataset path {dataset_path} not found!!")
        self.createDataset(self.out_path)

        # out_file = f"{self.out_path}.npz" if not self.out_path.endswith(".npz") else self.out_path
        # if not os.path.exists(out_file):
        #     raise FileNotFoundError(f"Processed dataset file not found: {out_file}")
        self.data = np.load(f"{self.out_path}.npz", mmap_mode='r')


    def createDataset(self, out_file):
        processed_files = []
        for file in os.listdir(self.dataset_path):
            audio_file = os.path.join(self.dataset_path, file)
            if os.path.isfile(audio_file):
                audio_data, _ = lr.load(audio_file, sr=self.sampling_rate, mono=self.mono)
                audio_data = quantization_u_law(audio_data)
                # process data (skipping right now)
                # --------------------------------#
                #                                 #
                #              TODO               # 
                #              TODO               #
                #                                 #
                # --------------------------------#
                processed_files.append(audio_data)
                self.sample_loc_util.append(self.sample_loc_util[-1] + (len(audio_data)-self.input_length))
        np.savez(out_file, *processed_files)
    
    def __len__(self):
        return self.sample_loc_util[-1]

    def __getitem__(self, idx):
        """
            args: 
            --- index (idx)

            --- locates the audio segment the index lies in and then the local index that is the index in that audio segment from where a sample 
            --- of inputs and targets will be retrieved 
        """
        # upper_bound=0
        data_idx = 0
        for i in range(1, len(self.sample_loc_util)):
            if idx < self.sample_loc_util[i]:
                data_idx=i-1
                break
        local_idx = idx - self.sample_loc_util[data_idx]
        
        audio_segment = self.data[f"arr_{data_idx}"]

        input_seq = audio_segment[local_idx : local_idx + self.input_length]
        target_seq = audio_segment[local_idx+self.input_length : local_idx+self.input_length + self.target_length]
        
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

def quantization_u_law(x):
    """
    quantizing 16 bit audio inputs to 0-255 range using meu law quantization
    assuming the audio input is already normalized in range [-1, 1]
    """
    x = np.array(x)
    u = 255
    mu_law_output = np.sign(x) * (np.log(1 + u*np.abs(x)))/(np.log(1+u))
    quantized = ((mu_law_output + 1)/2 * 255).astype(np.uint8)
    return torch.tensor(quantized)