# Author: Yael Segal
import torch
import os
import wandb
import logging
from torch.utils.data.sampler import Sampler
import torch.distributed as distributed
import math

class Logger(object):
    def __init__(self, args) -> None:
        self.args = args
        if args.wandb:
            dir_to_log = os.path.dirname(os.path.dirname(__file__))
            wandb.init(project="eend",entity='mlspeech',config=vars(args),settings=wandb.Settings(code_dir=dir_to_log))
        else:
            formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
            logging.basicConfig(level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
            self.logger = logging.getLogger("Pytorch")
            fh = logging.FileHandler(args.model_save_dir + "/train.log", mode='w')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            # ===================================================================
            self.logger.info(str(args))

    def info(self, msg_dict):

        if self.args.wandb:
            if type(msg_dict) == dict:
                wandb.log(msg_dict)
            else:
                print(msg_dict)
        else:
            if type(msg_dict) == dict:
                msg = ""
                for k, v in msg_dict.items():
                    msg += f"{k}: {v}, "
                self.logger.info(msg)
            else:
                self.logger.info(msg_dict)

class DistributedSubsetSampler(Sampler):
    
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, part=1):
        if num_replicas is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = distributed.get_world_size()
        if rank is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        selected_num_samples = len(self.dataset) * part
        self.num_samples = int(math.ceil(selected_num_samples * 1.0 / self.num_replicas))
        self.indices = list(range(len(dataset)))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    
    def __iter__(self):


        torch.manual_seed(self.epoch)
        if self.shuffle:
            all_indices = torch.randperm(len(self.indices))
        else:
            all_indices = self.indices.copy()
        indices = all_indices[:self.total_size]

        # add extra samples to make it evenly divisible
        if (self.total_size - len(indices)) > 0:
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_mask_by_len(y_lens, device):
    
    max_len = max(y_lens)
    mask_tensor = torch.ones(size=(len(y_lens),max_len))
    for idx, target_len in enumerate(y_lens):
        mask_tensor[idx, target_len:] = 0
    return mask_tensor.to(device)

def is_best_measure(measure, best_measure, m_type):
    if m_type == "min":
        if measure < best_measure:
            return True
    elif m_type == "max":
        if measure > best_measure:
            return True