
import argparse
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = "/".join(currentdir.split("/")[:-1])
sys.path.insert(0, parentdir) 
import kaldi_data
import feature
import torch
import tqdm


parser = argparse.ArgumentParser(description='EEND training')

parser.add_argument('--data_dir', default="/home/mlspeech/shua/home/Shua/recipies/Diar/EEND/egs/callhome/v1_fisher/data/simu_100k/data/swb_sre_cv_ns1n2n3n4_beta2n2n5n9_500", type=str,
                    help='kaldi-style data dir used for training.')
parser.add_argument('--output_dir',  default="/home/data/segalya/EEND_data", type=str,
                    help='directory to save the features in.')
parser.add_argument('--rate', default=8000, type=int,
                    help='sampeling rate')
parser.add_argument('--frame_size', default=200, type=int,
                    help='sampeling rate')
parser.add_argument('--frame_shift', default=80, type=int,
                    help='sampeling rate')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size')
parser.add_argument('--num_workers', default=16, type=int,
                    help='num_workers ')
args = parser.parse_args()


def my_collate(batch):
    data, target = list(zip(*batch))
    return [data, target]
class CalcSpec(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args
        self.data = kaldi_data.KaldiData(args.data_dir)
        self.idx2wav = {idx: wav for idx, wav in enumerate(self.data.wavs)}


    def __getitem__(self, index):
        rec = self.idx2wav[index]
        data_len = int(self.data.reco2dur[rec] * self.args.rate / self.args.frame_shift)
        data, rate = self.data.load_wav(rec, 0 , data_len * self.args.frame_shift) 
        Y = feature.stft(data, self.args.frame_size, self.args.frame_shift)
        path_array = self.data.wavs[rec].split("/")
        base_folder = os.path.join(self.args.output_dir, "npy", *path_array[-3:-1])
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        new_filename = os.path.join(base_folder, path_array[-1].replace(".wav", ".npy") )
        np.save(new_filename, Y)
        return rec, new_filename
    def __len__(self):
        return len(self.idx2wav)
        
dataset  = CalcSpec(args)  
dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate,
                        num_workers=args.num_workers, pin_memory=True)
rec_npy_dict = {}
for batch in tqdm.tqdm(dataloader):
    recs, npys = batch
    for rec, npy in zip(recs, npys):
        rec_npy_dict[rec] = npy
new_dir = os.path.join(args.output_dir, "data", os.path.basename(args.data_dir))
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
with open(os.path.join(new_dir , "rec2npy.scp"), "w") as f:
    for rec in rec_npy_dict:
        f.write("{} {}\n".format(rec, rec_npy_dict[rec]))