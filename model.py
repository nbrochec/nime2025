import torch
import torch.nn as nn
import torchaudio.transforms as Taudio

class LogMelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=8000, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=50):
        super(LogMelSpectrogramLayer, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        
        self.mel_scale = Taudio.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min
        )

        self.amplitude_to_db = Taudio.AmplitudeToDB(stype='power')

    def min_max_normalize(self, t: torch.Tensor, min: float = 0.0, max: float = 1.0) -> torch.Tensor:
        min_tensor = torch.tensor(min, dtype=t.dtype, device=t.device)
        max_tensor = torch.tensor(max, dtype=t.dtype, device=t.device)
        eps = 1e-5
        t_min = torch.min(t)
        t_max = torch.max(t)

        if (t_max - t_min) == 0:
            t_std = (t - t_min) / ((t_max - t_min) + eps)
        else:
            t_std = (t - t_min) / (t_max - t_min)
        
        t_scaled = t_std * (max_tensor - min_tensor) + min_tensor
        return t_scaled

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        x = self.min_max_normalize(x)
        return x.to(torch.float32)
    
class CustomConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(CustomConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch = nn.BatchNorm2d(output_channels)
        self.activ = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        return x

class nime2025(nn.Module):
    def __init__(self, output_nbr, sr, classnames, segment_length):
        super(nime2025, self).__init__()

        self.sr = sr
        self.classnames = classnames
        self.seglen = segment_length

        self.logmel = LogMelSpectrogramLayer(sample_rate=self.sr)
        
        self.cnn = nn.Sequential(
            CustomConv2d(1, 64, (2,3), "same"),
            CustomConv2d(64, 64, (2,3), "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            CustomConv2d(64, 128, (2,3), "same"),
            CustomConv2d(128, 128, (2,3), "same"),
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.25),
            CustomConv2d(128, 256, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            CustomConv2d(256, 256, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            CustomConv2d(256, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            CustomConv2d(512, 512, 2, "same"),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25),
            CustomConv2d(512, 512, 2, "same"),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout1d(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout1d(0.25),
            nn.Linear(128, output_nbr),
        )

    @torch.jit.export
    def get_sr(self):
        return self.sr
    
    @torch.jit.export
    def get_classnames(self):
        return self.classnames
    
    @torch.jit.export
    def get_seglen(self):
        return self.seglen

    def forward(self, x):
        x = self.logmel(x)
        x = self.cnn(x)
        x_flat = x.view(x.size(0), -1)
        z = self.fc(x_flat)
        return z