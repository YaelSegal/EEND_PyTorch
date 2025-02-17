#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import torch
import numpy as np
from eend import kaldi_data
from eend import feature
import timeit, functools
import torchaudio
import tqdm
import time
import math
def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def my_collate(batch):
    data, target = list(zip(*batch))
    return [data, target]


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            raw_wav=False,
            model_sr = 16000,
            npy_dir = None,
            voice_only=False,
            ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        if n_speakers == -1:
            n_speakers = None
        self.n_speakers = n_speakers 
        self.chunk_indices = []
        self.label_delay = label_delay

        self.data = kaldi_data.KaldiData(self.data_dir, npy_dir=npy_dir)
        self.npy_dir = npy_dir
        self.model_sr = model_sr
        self.raw_wav = raw_wav
        self.resample = torchaudio.transforms.Resample(rate, model_sr)
        # nspeaker_dict = {0:0, 1:0, 2:0}

        # make chunk indices: filepath, start_frame, end_frame
        for rec in tqdm.tqdm(self.data.wavs):
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            if voice_only:
                st = 0
                ed = data_len
                start = st * self.subsampling
                end = ed * self.subsampling
                labels = feature.get_labels(self.data, rec, start, end, self.frame_size,self.frame_shift,None) # TODO- check speed of get info
                # split labels into voice chunks
                                # Find indices where the active status changes
                sum_labels = np.sum(labels,axis=1) 
                change_indices = np.where(np.diff( (sum_labels> 0)))[0] + 1

                # Split the array into segments based on the change indices
                segments = np.split(labels, change_indices)
                start = 0
                for seg in segments:
                    num_speakers = (np.sum(seg,axis=0) > 0).sum()
                    if seg[0][0] == 0 or self.n_speakers is not None and num_speakers > self.n_speakers:
                        start += seg.shape[0]
                        continue
                    seg_start = start 
                    seg_end = start + seg.shape[0]
                    if seg_end - seg_start > self.chunk_size:
                        num_new_segments = math.ceil((seg_end - seg_start) / self.chunk_size)
                        seg_shift = math.ceil((seg_end - seg_start) / num_new_segments)
                        while seg_end - seg_start > seg_shift:
                            self.chunk_indices.append(
                                (rec, seg_start , (seg_start + seg_shift) ))
                            seg_start += seg_shift
                    if seg_end - seg_start < 10:
                        continue
                    self.chunk_indices.append(
                        (rec, seg_start , seg_end ))
                    start += seg.shape[0]
            else:
                
                st = 0
                ed = data_len
                start = st * self.subsampling
                end = ed * self.subsampling
                labels_all = feature.get_labels(self.data, rec,start, end, self.frame_size,self.frame_shift,None)
                for st, ed in _gen_frame_indices(
                        data_len, chunk_size, chunk_size, use_last_samples,
                        label_delay=self.label_delay,
                        subsampling=self.subsampling):
                    start = st * self.subsampling
                    end = ed * self.subsampling

                    # labels = feature.get_labels(self.data, rec, start, end, self.frame_size,self.frame_shift,None) 
                    labels = labels_all[start:end] 
        
                    num_speakers = (np.sum(labels,axis=0) > 0).sum()
                    if self.n_speakers is not None and num_speakers > self.n_speakers:
                        continue
                        
                    self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))

        # if len(self.chunk_indices) > 1000:
        #     self.chunk_indices = self.chunk_indices[2050:2150]
        #     # self.chunk_indices = self.chunk_indices[:500]
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        if self.raw_wav:

            Y, wav_sr = feature.get_wav(self.data, rec, st, ed,self.frame_shift)
            Y_ss = torch.FloatTensor(Y)
            if wav_sr != self.model_sr:
                Y_ss = self.resample(Y_ss)
            T = feature.get_labels(self.data, rec, st, ed, self.frame_size,self.frame_shift,None)
            T_ss = feature.subsample(T, self.subsampling)
            T_ss = torch.FloatTensor(T_ss)
            return Y_ss, T_ss
        else:
            if self.npy_dir is not None:
                Y = feature.loadSTFT(self.data,rec,st,ed)
                T = feature.get_labels(self.data, rec, st, ed, self.frame_size,self.frame_shift,self.n_speakers)
            else:
                Y, T = feature.get_labeledSTFT(
                    self.data,
                    rec,
                    st,
                    ed,
                    self.frame_size,
                    self.frame_shift,
                    self.n_speakers)
            # Y: (frame, num_ceps)
            Y = feature.transform(Y, self.input_transform)
            # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
            Y_spliced = feature.splice(Y, self.context_size)
            # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
            Y_ss = feature.subsample(Y_spliced, self.subsampling)
            T_ss = feature.subsample(T, self.subsampling)

            Y_ss = torch.FloatTensor(Y_ss)
            T_ss = torch.FloatTensor(T_ss)
            return Y_ss, T_ss
