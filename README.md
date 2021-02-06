# ORCA-SPOT: An Automatic Killer Whale Sound Detection Toolkit Using Deep Learning

## General Description
ORCA-SPOT is a deep learning based alogrithm which was initially designed for killer whale sound detection in noise heavy underwater recordings. ORCA-SPOT distinguishes between two types of sounds: killer whales and noise (binary classification problem). It is based on a convolutional neural network architecture which is capable to segment large bioacoustic archives. ORCA-SPOT includes a data preprocessing pipeline plus the network architecture itself for training the model. For a detailed description about the core concepts, network architecture, preprocessing and evaluation pipeline please see our corresponding publication https://www.nature.com/articles/s41598-019-47335-w.


## General Information
We are currently in the process to publish a paper as a general guidance which allows everybody (background independent) to train, validate, and evaluate their own target species, by using this deep learning framework, in order to segment/detect valuable signals within noisy bioacoustic recordings. Goal is to provide the community with an animal independent sound segementation toolkit to label their data. Once the guidline is finished we will announce and link it also here.

## Reference
If ORCA-SPOT is used for your own research please cite the following publication: ORCA-SPOT: An Automatic Killer Whale Sound Detection Toolkit Using Deep Learning (https://www.nature.com/articles/s41598-019-47335-w)

```
@article{bergler:2019,
author = {Bergler, Christian and Schröter, Hendrik and Cheng, Rachael Xi and Barth, Volker and Weber, Michael and Noeth, Elmar and Hofer, Heribert and Maier, Andreas},
year = {2019},
month = {12},
pages = {},
title = {ORCA-SPOT: An Automatic Killer Whale Sound Detection Toolkit Using Deep Learning},
volume = {9},
journal = {Scientific Reports},
doi = {10.1038/s41598-019-47335-w}
}
```
## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

## Python, Python Libraries, and Version
ORCA-SPOT is a deep learning algorithm which was implemented in Python (Version=3.6) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.3.0, TorchVision=0.4.1). Moreover it requires the following Python libraries: Pillow, MatplotLib, Librosa, Soundfile, TensorboardX (recent versions).
ORCA-SPOT is currently compatible with PyTorch (Version=1.7.1, TorchVision=0.8.2, TorchAudio=0.7.2)

## Required Filename Structure for Training
In order to properly load and preprocess your data to train the network you need to prepare the filenames of your audio data clips to fit the following template/format:

Filename Template: call/noise-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav

1st-Element: call/noise-XXX = call-XXX or noise-XXX for pointing out wether it is a target signal or any kind of noise signal. XXX is a placeholder for any kind of string which could be added for more specific label information, e.g. call-N9, noise-boat

2nd-Element: ID = unique ID (natural number) to identify the audio clip

3rd-Element: YEAR = year of the tape when it has been recorded

4th-Element: TAPENAME = name of the recorded tape (has to be unique in order to do a proper data split into train, devel, test set by putting one tape only in only one of the three sets

5th-Element: STARTTIME = start time of the audio clip in milliseconds with respect to the original recording (natural number)

6th-Element: ENDTIME = end time of the audio clip in milliseconds with respect to the original recording(natural number)

Due to the fact that the underscore (_) symbol was chosen as a delimiter between the single filename elements please do not use this symbol within your filename except for separation.

Examples of valid filenames:

call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919

Label Name=call-Orca-A12, ID=929, Year=2019, Tapename=Rec-031-2018-10-19-06-59-59-ASWMUX231648, Starttime in ms=2949326, Starttime in ms=2949919

noise-humanVoice_2381_2010_101BC_149817_150055.wav

Label Name=noise-humanVoice, ID=2381, Year=2010, Tapename=101BC, Starttime in ms=149817, Starttime in ms=150055

## Required Directory Structure for Training
ORCA-SPOT does its own training, validation, and test split of the entire provided data archive. The entire data could be either stored in one single folder and ORCA-SPOT will generate the datasplit by creating three CSV files (train.csv, val.csv, and test.csv) representing the partitions and containing the filenames. There is also the possibility to have a main data folder and subfolders containing special type of files e.g. N9_calls, N1_calls, unknown_orca_calls, boat_noise, human_noise, other_species_noise. ORCA-SPOT will create for each subfolder a stand-alone train/validation/test split and merges all files of each subfolder partition together. ORCA-SPOT ensures that no audio files of a single tape are spread over training/validation/testing. Therefore it moves all files of one tape into only one of the three partitions. If there is only data from one tape or if one of the three partitions do not contain any files the training will not be started. By default ORCA-SPOT uses 70% of the files for training, 15% for validation, and 15% for testing. In order to guarantee such a distribution it is important to have a similar amount of labeled files per tape.

## Network Training
For a detailed description about each possible training option we refer to the usage/code in main.py (usage: main.py -h). This is just an example command in order to start network training:

```main.py --debug --augmentation 1 --max_train_epochs 100 --noise_dir noise_dir --resnet 18 --lr 10e-5 --max_pool 2 --start_from_scratch --conv_kernel_size 7  --batch_size 16 --num_workers 8 --data_dir data_dir --cache_dir cache_dir --model_dir model_dir --log_dir log_dir --checkpoint_dir checkpoint_dir --summary_dir summary_dir --n_fft 4096 --hop_length 441 --freq_compression linear```

## Network Testing and Evaluation
During training ORCA-SPOT will be verified on an independent validation set. In addition ORCA-SPOT will be automatically evaluated on the test set. In both cases multiple machine learning metrics (loss, accuracy, recall, f1-score, confusion matrix, etc.) will be calculated and documented. In addition it will store spectrograms of correct classifications (true positive/negative) as well as misclassifications (false positive/negative)  within the training/validiation/test set. All documented results and the entire training process could be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir /directory_to_model/summaries/```

There exist also the possibility to evaluate your model on a entire tape. The prediction script (predict.py) implements a sliding window approach in order to feed the trained network  with audio slices of a given sequence length and a give hop size. According to the chosen threshold  (network confidence) the network classifies a single audio slice as "target_signal" or "noise". For a detailed description about each possible option we refer to the usage/code in predict.py (usage: predict.py -h). This is just an example command in order to start the prediction:

Example Command:

```predict.py -d --model_path model_dir/model.pk --log_dir log_dir --sequence_len 2 --hop 0.5 --threshold 0.75 --num_workers 8 --no_cuda audio_tape.wav```

