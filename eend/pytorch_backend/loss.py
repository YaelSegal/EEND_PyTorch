# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import permutations
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import time
from  multiprocessing.pool import ThreadPool
import signal
"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""


def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    label_perms = [label[..., list(p)] for p
                    in permutations(range(label.shape[-1]))]
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(
            pred[label_delay:, ...],
            l[:len(l) - label_delay, ...]) for l in label_perms])
    min_loss = losses.min() * (len(label) - label_delay)
    min_index = losses.argmin().detach()
    
    return min_loss, label_perms[min_index]


def batch_pit_loss(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    loss_w_labels = [pit_loss(y, t, label_delay)
                     for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss, labels

def batch_pit_n_speaker_loss(ys, ts, n_speakers_list, mask):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    device = ys[0].device
    max_n_speakers = max(n_speakers_list)
    t_speakers_padd = [torch.cat((ti, torch.zeros((ti.shape[0], max_n_speakers-n_speakers_list[i])).to(device)),axis=1) for i, ti in enumerate(ts)]
    mask = mask.unsqueeze(2).repeat( (1,1,max_n_speakers))
    # (B, T, C)
    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = [torch.roll(t, -shift, dims=1) for t in t_speakers_padd]
        targets = torch.nn.utils.rnn.pad_sequence(ts_roll, batch_first=True)
        min_seq_len = min(ys.shape[1],targets.shape[1]) 
        # loss: (B, T, C)
        loss = F.binary_cross_entropy_with_logits(ys[:,:min_seq_len,:max_n_speakers ], targets[:,:min_seq_len], reduction='none')
        # sum over time: (B, C)
        loss = torch.sum(loss * mask,axis=1 )
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, axis=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    perms = torch.tensor(
        list(permutations(range(max_n_speakers))),
        dtype=torch.int32,
    )
    # y_ind: [0,1,2,3]
    y_ind = torch.arange(max_n_speakers, dtype=torch.int32)
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = torch.remainder(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind, t_ind], axis=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, axis=1)

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]
    masks = torch.full_like(losses_perm, torch.inf)
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, axis=1)[0])
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = torch.argmin(losses_perm, axis=1)

    labels_perm = [t[:, perms[idx]] for t, idx in zip(t_speakers_padd, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm

def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    label = label[:len(label) - label_delay, ...]
    decisions = torch.sigmoid(pred[label_delay:, ...]) > 0.5
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum() / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    return res

def report_diarization_error(ys, labels):
    """
    Reports diarization errors
    Should be called with torch.no_grad

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_avg = {}
    cnt = 0
    for y, t in zip(ys, labels):
        y_len = y.shape[0]
        t_len = t.shape[0]
        select_len = min(y_len, t_len)
        y_speakers = y.shape[1]
        t_speakers = t.shape[1]
        selected_speakers = min(y_speakers, t_speakers)
        stats = calc_diarization_error(y[:select_len, :selected_speakers], t[:select_len, :selected_speakers])
        for k, v in stats.items():
            stats_avg[k] = stats_avg.get(k, 0) + float(v)
        cnt += 1
    
    stats_avg = {k:v/cnt for k,v in stats_avg.items()}
    return stats_avg
        


def speakers_loss(prob, n_speakers, device):
    labels = torch.concat([torch.LongTensor([[1] * n_spk + [0]]) for n_spk in n_speakers], axis=1).to(device)
    selected_prob = torch.concat([(logiti[:n_speakers[i]+1, :]).reshape(-1) for i, logiti in enumerate(prob)])
    loss = F.binary_cross_entropy(selected_prob, labels.squeeze(0).float())
    return loss

class PitLoss(nn.Module):
    # permutation
    def __init__(self, weight=None, reduction="mean"):
        super(PitLoss, self).__init__()
        self.eps = 1e-14
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight,reduction="none" )

    def forward(self, pred_frames_out, targets,  mask=None): 
        device = pred_frames_out.device
        n_speakers = [ti.shape[1] for ti in targets] 
        max_speaker = max(n_speakers)
        t_speakers_padd = [torch.cat((ti, torch.zeros((ti.shape[0], max_speaker-n_speakers[i])).to(device)),axis=1) for i, ti in enumerate(targets)]
        targets = torch.nn.utils.rnn.pad_sequence(t_speakers_padd, batch_first=True)
        batch_size, seq_len, targets_type = targets.shape


        per_list_values = [ i for i in permutations(range(targets_type))]

        per_list_loss = []
        for target_indexes in per_list_values:
            current_target = targets[:,:, target_indexes]
            speaker_value_list = []

            for speaker_idx in range(targets_type):
                speaker_target = current_target[:,:,speaker_idx]
                speaker_pred = pred_frames_out[:,: ,speaker_idx]

                select_seq_len = min(seq_len, speaker_pred.shape[1])
                total_speaker_loss = self.bce_loss(speaker_pred[:, :select_seq_len], speaker_target[:, :select_seq_len])

                if mask is None:
                    speaker_value_list.append(torch.mean(total_speaker_loss,axis=1))
                else :
                    speaker_value_list.append(torch.sum(total_speaker_loss * mask[:, :select_seq_len],axis=1 )/ (torch.sum(mask,axis=1) + self.eps))
  
            current_target_loss =  torch.vstack(speaker_value_list)

            per_list_loss.append(torch.mean(current_target_loss, axis=0)) # calculate mean speakers loss for each example, ,TODO- maybe mean by example speakers number
        
        best_val =  torch.min(torch.vstack(per_list_loss), dim =0)
        selected_per_loss =  best_val[0]
        return selected_per_loss.mean(), best_val[1]
            
 
def create_speaker_segments(preds, total_speakers_num):
    
    active_speakers_array = np.zeros((total_speakers_num,len(preds)))
    if type(preds) is list:
        if len(preds) == 0:
            return []
        for idx, speaker_list in enumerate(preds):
            for speaker in speaker_list:
                active_speakers_array[int(speaker), idx] = 1
    else:
        active_speakers_array = preds.T
        if active_speakers_array.sum() == 0:
            return []

    segments_array = []

    for speaker_idx in range(len(active_speakers_array)):
        speaker_segments = []
        current_speaker = active_speakers_array[speaker_idx]
        speaker_diff = np.diff(current_speaker)
        speaker_changed = np.where(speaker_diff!=0)[0]
        prev_active = -1
        for changed_idx in speaker_changed:
            if current_speaker[changed_idx] == 1:
                start = prev_active if prev_active>0 else 0
                speaker_segments.append([start, changed_idx+1, speaker_idx])

            else: # current_speaker[changed_idx] == 0
                prev_active = changed_idx + 1
        if current_speaker[-1] ==1:
            start = prev_active if prev_active>0 else 0
            speaker_segments.append([start, len(current_speaker), speaker_idx])
        segments_array.extend(speaker_segments)
    return segments_array

def create_pyannote_annotation(segments_array_sorted, window):
    
    current_annotation = Annotation()
    for segment in segments_array_sorted:
        start, end, speaker = segment
        start_sec = round(start* window, 3)
        end_sec = round(end * window, 3)
        speaker = str(speaker)
        current_annotation[Segment(start_sec, end_sec)] = speaker

    return current_annotation

class DiarizationErrorPyannote():
    def __init__(self, collar=0, skip_overlap=True, threshold=0.5, window=0.1) -> None:
        self.metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap )
        self.threshold = threshold
        self.window = window

    def __call__(self, hypothesis_tensor, reference_tensor, lens=None) -> Any:
        n_speakers = [t.shape[1] for t in reference_tensor]
        for idx, (example_pred_logits, example_target) in enumerate(zip(hypothesis_tensor, reference_tensor)):
            n_spk = n_speakers[idx]
            if lens is not None:
                example_pred_logits = example_pred_logits[:lens[idx]]
                example_target = example_target[:lens[idx]]
            example_pred_logits = example_pred_logits[:, :n_spk] # remove last speaker
            example_pred = torch.sigmoid(example_pred_logits)
            example_pred = example_pred > self.threshold

            pred_speaker_segments = create_speaker_segments(example_pred.cpu().numpy(), n_spk)
            target_speaker_segments = create_speaker_segments(example_target.cpu().numpy(), n_spk)
            pred_segments_array_sorted = sorted(pred_speaker_segments,key =lambda x:x[0])
            tagret_segments_array_sorted = sorted(target_speaker_segments,key =lambda x:x[0])
            hypothesis = create_pyannote_annotation(pred_segments_array_sorted, self.window)
            reference = create_pyannote_annotation(tagret_segments_array_sorted, self.window)
            example_der = self.metric(reference, hypothesis, detailed=True)

        return self.metric[:]

    def compute(self):
        states = self.metric[:]
        states["DER"] = abs(self.metric)
        return states
    
    def reset(self):
        self.metric.reset()

def init_pool(batch_size=2):
    
   original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
#    pool = multiprocessing.pool(batch_size)
   pool = ThreadPool(batch_size)
   signal.signal(signal.SIGINT, original_sigint_handler)
   return pool

class DiarizationErrorPyannoteProcess():

    def __init__(self, collar=0, skip_overlap=True, threshold=0.5, window=0.1) -> None:
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.threshold = threshold
        self.window = window
        self.total_der_dict = {}

    def calc_example_der(self, example_idx,example_pred, example_target, n_spk, example_len):

        example_pred = example_pred[:example_len]
        example_target = example_target[:example_len]
        example_pred = example_pred[:, :n_spk] # remove last speaker
        example_pred = example_pred > self.threshold

        pred_speaker_segments = create_speaker_segments(example_pred, n_spk)
        target_speaker_segments = create_speaker_segments(example_target, n_spk)
        pred_segments_array_sorted = sorted(pred_speaker_segments,key =lambda x:x[0])
        tagret_segments_array_sorted = sorted(target_speaker_segments,key =lambda x:x[0])
        hypothesis = create_pyannote_annotation(pred_segments_array_sorted, self.window)
        reference = create_pyannote_annotation(tagret_segments_array_sorted, self.window)
        metric_der = DiarizationErrorRate(collar=self.collar, skip_overlap=self.skip_overlap)
        example_der = metric_der(reference, hypothesis, detailed=True)
        return example_der

    def __call__(self, hypothesis_tensor, reference_tensor, lens=None) -> Any:
        n_speakers = [t.shape[1] for t in reference_tensor]
        batch_size = len(n_speakers)
        num_process = max(16,int(batch_size/2),1)
        # num_process = 1
        if lens is None:
            lens = [len(t) for t in reference_tensor]
        pool = init_pool(num_process)
        der_dict = {}
        list_idx = list(range(0,batch_size))
        for i in range(0, batch_size, num_process):
            try:
                selected_idx_batch = list_idx[i:i+num_process]

                params_list = [ (example_idx,  torch.sigmoid(hypothesis_tensor[example_idx]).cpu().numpy(), reference_tensor[example_idx].cpu().numpy(),
                                 n_speakers[example_idx], lens[example_idx]) for example_idx in selected_idx_batch]
                
                example_der_dict_list = pool.starmap(self.calc_example_der, (params_list))
                for example_der_dict in example_der_dict_list:
                    for k, v in example_der_dict.items():
                        der_dict[k] = der_dict.get(k, 0) + v
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
            except Exception as e:
                print(e)
                pool.close()
        pool.close()
        pool.join()

        for k, v in der_dict.items():
            self.total_der_dict[k] = self.total_der_dict.get(k, 0) + v
        return der_dict

    def reset(self):
        self.total_der_dict = {}


    def compute(self):
        states = self.total_der_dict
        states["DER"] = (states['false alarm']  + states['missed detection'] + states['confusion'])/ states['total']
        return states

from torchmetrics import Metric
from torch import Tensor
class DiarizationErrorPyannoteProcessMetric(Metric):
    
    def __init__(self, collar=0, skip_overlap=True, threshold=0.5, window=0.1) -> None:
        super().__init__()
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.threshold = threshold
        self.window = window
        self.add_state("confusion", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("missed_detection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_alarm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")


    def calc_example_der(self, example_idx,example_pred, example_target, n_spk, example_len):

        example_pred = example_pred[:example_len]
        example_target = example_target[:example_len]
        example_pred = example_pred[:, :n_spk] # remove last speaker
        example_pred = example_pred > self.threshold

        pred_speaker_segments = create_speaker_segments(example_pred, n_spk)
        target_speaker_segments = create_speaker_segments(example_target, n_spk)
        pred_segments_array_sorted = sorted(pred_speaker_segments,key =lambda x:x[0])
        tagret_segments_array_sorted = sorted(target_speaker_segments,key =lambda x:x[0])
        hypothesis = create_pyannote_annotation(pred_segments_array_sorted, self.window)
        reference = create_pyannote_annotation(tagret_segments_array_sorted, self.window)
        metric_der = DiarizationErrorRate(collar=self.collar, skip_overlap=self.skip_overlap)
        example_der = metric_der(reference, hypothesis, detailed=True)
        return example_der
    
    def update(self, hypothesis_tensor: list, reference_tensor: list, lens=None, n_speakers=None):

        if n_speakers is None:
            n_speakers = [t.shape[1] for t in reference_tensor]

        batch_size = len(n_speakers)
        # num_process = max(16,int(batch_size/2),1)
        num_process = 8 
        # num_process = 1
        if lens is None:
            lens = [len(t) for t in reference_tensor]
        pool = init_pool(num_process)
        der_dict = {}
        list_idx = list(range(0,batch_size))
        for i in range(0, batch_size, num_process):
            try:
                selected_idx_batch = list_idx[i:i+num_process]

                params_list = [ (example_idx,  torch.sigmoid(hypothesis_tensor[example_idx]).cpu().numpy(), reference_tensor[example_idx].cpu().numpy(),
                                 n_speakers[example_idx], lens[example_idx]) for example_idx in selected_idx_batch]
                
                example_der_dict_list = pool.starmap(self.calc_example_der, (params_list))
                for example_der_dict in example_der_dict_list:
                    for k, v in example_der_dict.items():
                        der_dict[k] = der_dict.get(k, 0) + v
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
            except Exception as e:
                print(e)
                pool.close()
        pool.close()
        pool.join()

        for k, v in der_dict.items():
            if k == 'confusion':
                self.confusion += v
            elif k == 'missed detection':
                self.missed_detection += v
            elif k == 'false alarm':
                self.false_alarm += v
            elif k == 'correct':
                self.correct += v
            elif k == 'total':
                self.total += v
            

    def compute(self):
        states = {}
        states["DER"] = (self.confusion  + self.missed_detection + self.false_alarm)/ self.total
        states["confusion"] = self.confusion
        states["missed detection"] = self.missed_detection
        states["false alarm"] = self.false_alarm
        states["correct"] = self.correct
        states["total"] = self.total

        return torch.FloatTensor([v for v in states.values()]).to(self.device)