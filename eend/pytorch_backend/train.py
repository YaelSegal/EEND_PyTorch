# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import numpy as np
from tqdm import tqdm


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from eend.pytorch_backend.models import TransformerModel, TransformerEDADiarization, NoamScheduler
from eend.pytorch_backend.diarization_dataset import KaldiDiarizationDataset, my_collate
from eend.pytorch_backend.loss import batch_pit_loss, report_diarization_error, PitLoss, speakers_loss, DiarizationErrorPyannote,batch_pit_n_speaker_loss, DiarizationErrorPyannoteProcess
from eend.pytorch_backend.utils import Logger, build_mask_by_len, is_best_measure


def train(args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """

    # Logger settings====================================================
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    logger = Logger(args)
    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set = KaldiDiarizationDataset(
        data_dir=args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )
    dev_set = KaldiDiarizationDataset(
        data_dir=args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )

    # Prepare model
    Y, T = next(iter(train_set))
    
    if args.model_type == 'Transformer':
        model = TransformerModel(
                n_speakers=args.num_speakers,
                in_size=Y.shape[1],
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False
                )
    elif args.model_type == 'EDATransformer':
        model = TransformerEDADiarization( in_size=Y.shape[1],
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                attractor_decoder_dropout=args.attractor_decoder_dropout,
                attractor_encoder_dropout=args.attractor_encoder_dropout,
                has_pos=False, shuffle=args.shuffle)
        
    else:
        raise ValueError('Possible model_type is "Transformer"')
    
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
    if device.type == "cuda":
        model = nn.DataParallel(model, list(range(args.gpu)))
    model = model.to(device)

    logger.info('Prepared model')
    logger.info(model)


    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'noam':
        # for noam, lr refers to base_lr (i.e. scale), suggest lr=1.0
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(args.optimizer)

    # For noam, we use noam scheduler
    if args.optimizer == 'noam':
        scheduler = NoamScheduler(optimizer,
                                  args.hidden_size,
                                  warmup_steps=args.noam_warmup_steps)

    # Init/Resume
    if args.initmodel:

        logger.info(f"Load model from {args.initmodel}")
        model.load_state_dict(torch.load(args.initmodel))

    train_iter = DataLoader(
            train_set,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=8,#TODO: change to 16
            collate_fn=my_collate
            )

    dev_iter = DataLoader(
            dev_set,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=16,
            collate_fn=my_collate
            )

    # Training
    # y: feats, t: label
    # grad accumulation is according to: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    best_measure = float("inf") if args.stop_measure_type == "min" else -float("inf")
    patience_count = 0
    criterion = PitLoss()
    for epoch in range(1, args.max_epochs + 1):
        if patience_count >= args.patience:
            logger.info(f"Early stop at epoch {epoch}")
            break
        model.train()
        # zero grad here to accumualte gradient
        optimizer.zero_grad()
        loss_epoch = 0
        num_total = 0
        for step, (y, t) in tqdm(enumerate(train_iter), ncols=100, total=len(train_iter)):
            y = [yi.to(device) for yi in y]
            t = [ti.to(device) for ti in t]
            n_speakers = [ti.shape[1] for ti in t] 
            lens = [ti.shape[0] for ti in t]
            y_tensor = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)
            if args.model_type == 'Transformer':
                output = model(y_tensor, ilens=lens)
                total_loss, label = batch_pit_loss(output, t)
            elif  args.model_type == 'EDATransformer':
                
                output = model(y_tensor, n_speakers=n_speakers)
                logits, attractors_prob = output
                
                mask = build_mask_by_len(lens, device=device)
                # loss, label = criterion(logits, t, mask=mask)
                loss, label = batch_pit_n_speaker_loss(logits, t, n_speakers,mask)
                batch_speakers_loss = speakers_loss(attractors_prob, n_speakers, device)
                total_loss = loss + args.attractor_loss_ratio * batch_speakers_loss
            else:
                raise NotImplementedError
            # clear graph here
            total_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # noam should be updated on step-level
                if args.optimizer == 'noam':
                    scheduler.step()
                if args.gradclip > 0:
                    nn.utils.clip_grad_value_(model.parameters(), args.gradclip)
            loss_epoch += total_loss.item()
            num_total += 1
        loss_epoch /= num_total
        
        model.eval()
        val_loss = []
        
        with torch.no_grad():
            stats_avg = {}
            # pyannote_stats1 = DiarizationErrorPyannote(collar=args.collar,skip_overlap=args.skip_overlap,window=(args.frame_shift/args.sampling_rate)*args.subsampling)
            pyannote_stats = DiarizationErrorPyannoteProcess(collar=args.collar,skip_overlap=args.skip_overlap,window=(args.frame_shift/args.sampling_rate)*args.subsampling)
            cnt = 0
            for y, t in dev_iter:
                y = [yi.to(device) for yi in y]
                t = [ti.to(device) for ti in t]
                n_speakers = [ti.shape[1] for ti in t] 
                lens = [ti.shape[0] for ti in t]
                y_tensor = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)
                if args.model_type == 'Transformer':
                    output = model(y_tensor, ilens=lens)
                    total_loss, label = batch_pit_loss(output, t)
                    stats = report_diarization_error(output, label)
                    for k, v in stats.items():
                        stats_avg[k] = stats_avg.get(k, 0) + v
                    
                elif  args.model_type == 'EDATransformer':
                    n_speakers = [ti.shape[1] for ti in t] 
                    output = model(y_tensor, n_speakers=n_speakers)
                    logits, attractors_prob = output
                    lens = [ti.shape[0] for ti in t]
                    mask = build_mask_by_len(lens, device=device)
                    loss, label = batch_pit_n_speaker_loss(logits, t, n_speakers,mask)
                    batch_speakers_loss = speakers_loss(attractors_prob, n_speakers, device)
                    total_loss = loss + args.attractor_loss_ratio * batch_speakers_loss
                    pyannote_stats(logits,label, lens)
                    # stats = report_diarization_error(logits, label)
                    # for k, v in stats.items():
                    #     stats_avg[k] = stats_avg.get(k, 0) + v
                else:
                    raise NotImplementedError
                val_loss.append(total_loss.item())
                cnt += 1
            if args.model_type == 'EDATransformer':
                stats_avg = pyannote_stats.compute()
            else:
                stats_avg = {k:v/cnt for k,v in stats_avg.items()}
                stats_avg['DER'] = stats_avg['diarization_error'] / stats_avg['speaker_scored'] * 100
            for k in stats_avg.keys():
                stats_avg[k] = round(stats_avg[k], 2)
        if args.patience > 0:
            if args.stop_measure == "der":
                current_measure = stats_avg["DER"]
                is_best = is_best_measure(stats_avg["DER"], best_measure, args.stop_measure_type)
            elif args.stop_measure == "loss":
                current_measure = np.mean(val_loss)
                is_best = is_best_measure(np.mean(val_loss), best_measure, args.stop_measure_type)
            else:
                raise NotImplementedError

        else:
            is_best = True
        if is_best:
            best_measure = current_measure
            if args.patience > 0:
                patience_count = 0

            model_filename = os.path.join(args.model_save_dir, f"transformer{epoch}.th")
            torch.save(model.state_dict(), model_filename)
        else:
            patience_count += 1
        msg = {"epoch": epoch, "train/loss": loss_epoch, "val/loss":np.mean(val_loss), "model": model_filename, "LR": optimizer.param_groups[0]['lr']}
        for k,val in stats_avg.items():
            msg[f"val/{k}"] = val
        logger.info(msg)
        # logger.info(f"Epoch: {epoch:3d}, LR: {optimizer.param_groups[0]['lr']:.7f},\
            # Training Loss: {loss_epoch:.5f}, Dev Stats: {stats_avg}")

    logger.info('Finished!')

