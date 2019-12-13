"""
Module: summary.py
Authors: Christian Bergler, Hendrik Schroeter
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 12.12.2019
"""

import os
import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch

from PIL import Image, ImageDraw
from torchvision.utils import make_grid
from visualization.utils import spec2img

"""
Prepare given image data for tensorboard visualization
"""
def prepare_img(img, num_images=4, file_names=None):
    with torch.no_grad():
        if img.shape[0] == 0:
            raise ValueError("`img` must include at least 1 image.")

        if num_images < img.shape[0]:
            tmp = img[:num_images]
        else:
            tmp = img
        tmp = spec2img(tmp)

        if file_names is not None:
            tmp = tmp.permute(0, 3, 2, 1)
            for i in range(tmp.shape[0]):
                try:
                    pil = Image.fromarray(tmp[i].numpy(), mode="RGB")
                    draw = ImageDraw.Draw(pil)
                    draw.text(
                        (2, 2),
                        os.path.basename(file_names[i]),
                        (255, 255, 255),
                    )
                    tmp[i] = torch.from_numpy(np.asarray(pil))
                except TypeError:
                    pass
            tmp = tmp.permute(0, 3, 1, 2)

        tmp = make_grid(tmp, nrow=1)
        return tmp.numpy()

"""
Prepare audio data for a given input audio file
"""
def prepare_audio(file_names, sr=44100, data_dir=None, num_audios=4) -> torch.Tensor:
    with torch.no_grad():
        out = []
        for i in range(min(num_audios, len(file_names))):
            file_name = file_names[i]
            if data_dir is not None:
                file_name = os.path.join(data_dir, file_name)
                audio, _ = librosa.load(file_name, sr)
                out.append(torch.from_numpy(audio))
        out = torch.cat(out)
        return out.reshape((1, -1))

"""
Plot ROC curve based on TPR/FPR
"""
def roc_fig(tpr, fpr, auc):
    fig = plt.figure()
    plt.plot(fpr, tpr, label="AUC: {}".format(auc))
    plt.legend(markerscale=0)
    plt.title("ROC curve")
    return fig
