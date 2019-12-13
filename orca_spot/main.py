#!/usr/bin/env python3

"""
Module: main.py
Authors: Christian Bergler, Hendrik Schroeter
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 12.12.2019
"""

import os
import json
import math
import pathlib
import argparse
import utils.metrics as m

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    DatabaseCsvSplit,
    DefaultSpecDatasetOps,
    Dataset,
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict
from models.residual_encoder import DefaultEncoderOpts
from models.residual_encoder import ResidualEncoder as Encoder
from models.classifier import Classifier, DefaultClassifierOpts

parser = argparse.ArgumentParser()

"""
Convert string to boolean.
"""
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

""" Directory parameters """
parser.add_argument(
    "--data_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
    "--noise_dir",
    type=str,
    default=None,
    help="Path to a directory with noise files used for data augmentation.",
)

""" Training parameters """
parser.add_argument(
    "--start_from_scratch",
    dest="start_scratch",
    action="store_true",
    help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--max_train_epochs", type=int, default=500, help="The number of epochs to train for the classifier."
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for the adam optimizer."
)

parser.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

""" Input parameters """
parser.add_argument(
    "--filter_broken_audio", action="store_true", help="Filter by a minimum loudness using SoX (Sound exchange) toolkit (option could only be used if SoX is installed)."
)

parser.add_argument(
    "--sequence_len", type=int, default=1280, help="Sequence length in ms."
)

parser.add_argument(
    "--freq_compression",
    type=str,
    default="linear",
    help="Frequency compression to reduce GPU memory usage. "
    "Options: `'linear'` (default), '`mel`', `'mfcc'`",
)

parser.add_argument(
    "--n_freq_bins",
    type=int,
    default=256,
    help="Number of frequency bins after compression.",
)

parser.add_argument(
    "--n_fft",
    type=int,
    default=4096,
    help="FFT size.")

parser.add_argument(
    "--hop_length",
    type=int,
    default=441,
    help="FFT hop length.")

parser.add_argument(
    "--augmentation",
    type=str2bool,
    default=True,
    help="Whether to augment the input data. "
    "Validation and test data will not be augmented.",
)

""" Network parameters """
parser.add_argument(
    "--resnet", dest="resnet_size", type=int, default=18, help="ResNet size"
)

parser.add_argument(
    "--conv_kernel_size", nargs="*", type=int, help="Initial convolution kernel size."
)

parser.add_argument(
    "--max_pool",
    type=int,
    default=None,
    help="Use max pooling after the initial convolution layer.",
)

parser.add_argument(
    "--latent_dim", type=int, default=1024, help="Dimensionality to reduce to."
)

ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

if ARGS.conv_kernel_size is not None and len(ARGS.conv_kernel_size):
    ARGS.conv_kernel_size = ARGS.conv_kernel_size[0]

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
"""
def get_audio_files():
    audio_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))
        if ARGS.filter_broken_audio:
            data_dir_ = pathlib.Path(ARGS.data_dir)
            audio_files = get_audio_files_from_dir(ARGS.data_dir)
            log.debug("Moving possibly broken audio files to .bkp:")
            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
            for f in broken_files:
                log.debug(f)
                bkp_dir = data_dir_.joinpath(f).parent.joinpath(".bkp")
                bkp_dir.mkdir(exist_ok=True)
                f = pathlib.Path(f)
                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.close()
            exit(1)
    return audio_files

"""
Save the trained model and corresponding options.
"""
def save_model(encoder, encoderOpts, classifier, classifierOpts, dataOpts, path):
    encoder = encoder.cpu()
    classifier = classifier.cpu()
    encoder_state_dict = encoder.state_dict()
    classifier_state_dict = classifier.state_dict()

    save_dict = {
        "encoderOpts": encoderOpts,
        "classifierOpts": classifierOpts,
        "dataOpts": dataOpts,
        "encoderState": encoder_state_dict,
        "classifierState": classifier_state_dict,
    }
    if not os.path.isdir(ARGS.model_dir):
        os.makedirs(ARGS.model_dir)
    torch.save(save_dict, path)

"""
Main function to compute data preprocessing, network training, evaluation, and saving.
"""
if __name__ == "__main__":

    encoderOpts = DefaultEncoderOpts
    classifierOpts = DefaultClassifierOpts
    dataOpts = DefaultSpecDatasetOps

    for arg, value in vars(ARGS).items():
        if arg in encoderOpts and value is not None:
            encoderOpts[arg] = value
        if arg in classifierOpts and value is not None:
            classifierOpts[arg] = value
        if arg in dataOpts and value is not None:
            dataOpts[arg] = value

    ARGS.lr *= ARGS.batch_size

    patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)

    patience_lr = int(max(1, patience_lr))

    log.debug("dataOpts: " + json.dumps(dataOpts, indent=4))

    sequence_len = int(
        float(ARGS.sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"]
    )
    log.debug("Training with sequence length: {}".format(sequence_len))
    input_shape = (ARGS.batch_size, 1, dataOpts["n_freq_bins"], sequence_len)

    log.info("Setting up model")

    encoder = Encoder(encoderOpts)
    log.debug("Encoder: " + str(encoder))
    encoder_out_ch = 512 * encoder.block_type.expansion

    classifierOpts["num_classes"] = 2
    classifierOpts["input_channels"] = encoder_out_ch
    classifier = Classifier(classifierOpts)
    log.debug("Classifier: " + str(classifier))

    split_fracs = {"train": .7, "val": .15, "test": .15}
    input_data = DatabaseCsvSplit(
        split_fracs, working_dir=ARGS.data_dir, split_per_dir=True
    )

    audio_files = get_audio_files()

    if ARGS.noise_dir:
        noise_files = [str(p) for p in pathlib.Path(ARGS.noise_dir).glob("*.wav")]
    else:
        noise_files = []

    datasets = {
        split: Dataset(
            file_names=input_data.load(split, audio_files),
            working_dir=ARGS.data_dir,
            cache_dir=ARGS.cache_dir,
            sr=dataOpts["sr"],
            n_fft=dataOpts["n_fft"],
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            freq_compression=ARGS.freq_compression,
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            seq_len=sequence_len,
            augmentation=ARGS.augmentation if split == "train" else False,
            noise_files=noise_files,
            dataset_name=split,
        )
        for split in split_fracs.keys()
    }

    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=ARGS.batch_size,
            shuffle=True,
            num_workers=ARGS.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        for split in split_fracs.keys()
    }

    model = nn.Sequential(
        OrderedDict([("encoder", encoder), ("classifier", classifier)])
    )
    trainer = Trainer(
        model=model,
        logger=log,
        prefix="classifier",
        checkpoint_dir=ARGS.checkpoint_dir,
        summary_dir=ARGS.summary_dir,
        n_summaries=4,
        start_scratch=ARGS.start_scratch,
    )

    metrics = {
        "tp": m.TruePositives(ARGS.device),
        "tn": m.TrueNegatives(ARGS.device),
        "fp": m.FalsePositives(ARGS.device),
        "fn": m.FalseNegatives(ARGS.device),
        "accuracy": m.Accuracy(ARGS.device),
        "f1": m.F1Score(ARGS.device),
        "precision": m.Precision(ARGS.device),
        "recall": m.Recall(ARGS.device),
        "TPR": m.TPR(ARGS.device),
        "FPR": m.FPR(ARGS.device),
    }

    optimizer = optim.Adam(
        model.parameters(), lr=ARGS.lr, betas=(ARGS.beta1, 0.999)
    )

    metric_mode = "max"
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        patience=patience_lr,
        factor=ARGS.lr_decay_factor,
        threshold=1e-3,
        threshold_mode="abs",
    )

    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=lr_scheduler,
        n_epochs=ARGS.max_train_epochs,
        val_interval=ARGS.epochs_per_eval,
        patience_early_stopping=ARGS.early_stopping_patience_epochs,
        device=ARGS.device,
        metrics=metrics,
        val_metric="accuracy",
        val_metric_mode=metric_mode,
    )

    encoder = model.encoder

    classifier = model.classifier

    path = os.path.join(ARGS.model_dir, "model.pk")

    save_model(
        encoder, encoderOpts, classifier, classifierOpts, dataOpts, path
    )

    log.close()
