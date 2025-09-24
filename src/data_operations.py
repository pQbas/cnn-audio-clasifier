from src.dataset import ESC50Dataset
import torch.nn as nn
import torchaudio.transforms as T


train_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=11025,
    ),
    T.AmplitudeToDB(),
    T.FrequencyMasking(freq_mask_param=30),
    T.TimeMasking(time_mask_param=80),
)

val_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=11025,
    ),
    T.AmplitudeToDB(),
)
