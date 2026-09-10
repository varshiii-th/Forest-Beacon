"""
model.py — PyTorch inference wrapper
Loads your trained .pth model and runs classification on audio files.
Plug in your own preprocessing if it differs from the default below.
"""
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

import torchaudio.transforms as T

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Model architecture definitions (must match training code exactly)  #
# ------------------------------------------------------------------ #
NUM_CLASSES = 2

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class CustomCNN(nn.Module):
    """4-block CNN: (B,1,64,T) → (B,2)"""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32),  nn.MaxPool2d(2),
            ConvBlock(32, 64),  nn.MaxPool2d(2),
            ConvBlock(64, 128), nn.MaxPool2d(2),
            ConvBlock(128,256), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256*4*4, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.pool(self.features(x))
        return self.head(x.view(x.size(0), -1))


class ResBlock(nn.Module):
    def __init__(self, ch, p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Dropout2d(p),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(x + self.block(x))


class AudioResNet(nn.Module):
    """ResNet-18-style for 1-channel spectrograms: (B,1,64,T) → (B,2)"""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        def layer(ic, oc, n, s=2):
            return nn.Sequential(
                nn.Sequential(nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
                              nn.BatchNorm2d(oc), nn.ReLU(inplace=True)),
                *[ResBlock(oc) for _ in range(n-1)]
            )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            layer(32,  64,  2, s=1),
            layer(64,  128, 2),
            layer(128, 256, 2),
            layer(256, 512, 2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.net(x)
        return self.head(x.view(x.size(0), -1))


# ------------------------------------------------------------------ #
#  Config (override via env vars)                                      #
# ------------------------------------------------------------------ #
SAMPLE_RATE   = int(os.getenv("MODEL_SAMPLE_RATE", "12000"))
N_MELS        = int(os.getenv("MODEL_N_MELS", "64"))
N_FFT         = int(os.getenv("MODEL_N_FFT", "512"))
HOP_LENGTH    = int(os.getenv("MODEL_HOP_LENGTH", "128"))
MAX_DURATION  = int(os.getenv("MODEL_MAX_DURATION_SEC", "60"))  # seconds
POSITIVE_IDX  = int(os.getenv("MODEL_POSITIVE_CLASS_IDX", "0"))


class SoundClassifier:
    """
    Thin wrapper around any torchscript / state-dict model.

    Expected model behaviour
    ------------------------
    Input  : (B, 1, N_MELS, T) float tensor (log-mel spectrogram)
    Output : (B, num_classes) logits  —  softmax applied here
    """

    def __init__(self, model_path: str, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"[Model] Using device: {self.device}")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Inject architecture classes into __main__ so pickle can find them.
        # The model was saved in a Colab notebook where classes lived in __main__,
        # so unpickling in any other script fails unless we patch this.
        import sys
        _main = sys.modules["__main__"]
        for _cls in [ConvBlock, CustomCNN, ResBlock, AudioResNet]:
            setattr(_main, _cls.__name__, _cls)

        # Try TorchScript first, fall back to full pickle load
        try:
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            logger.info("[Model] Loaded as TorchScript")
        except Exception:
            self.model = torch.load(str(model_path), map_location=self.device, weights_only=False)
            logger.info("[Model] Loaded as full model (pickle)")

        self.model.eval()
        self.model.to(self.device)

        # Build reusable transforms
        self._mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=50,
            f_max=6000,
        ).to(self.device)

        self._amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        logger.info("[Model] Ready ✓")

    # ---------------------------------------------------------------- #

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        """Load audio file → mono → resampled waveform tensor."""
        import soundfile as sf
        
        data, sr=sf.read(audio_path)

        # convert to torch tensor
        waveform=torch.tensor(data).float()

        #ensure shape=(1,N)
        if waveform.ndim==1:
            waveform=waveform.unsqueeze(0)
        else:
            waveform=waveform.T
        
        # convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        return waveform
    

         


    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """(1, N) waveform → (1, 1, M, T) log-mel tensor."""
        chunk_samples = SAMPLE_RATE * MAX_DURATION
        # Pad if shorter than one chunk
        if waveform.shape[1] < chunk_samples:
            pad = chunk_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        waveform = waveform.to(self.device)
        mel    = self._mel(waveform)        # (1, M, T)
        mel_db = self._amp_to_db(mel)
        #mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return mel_db.unsqueeze(0)          # (1, 1, M, T)

    def predict(self, audio_path: str) -> dict:
        """
        Sliding-window inference over the full audio clip.

        The clip is split into MAX_DURATION-second windows with 50% overlap.
        Any window classified as positive → whole clip is positive.
        Reports the highest positive confidence seen across all windows.

        Returns
        -------
        {
            "is_positive":    bool,
            "confidence":     float,   # highest positive window confidence
            "positive_windows": int,   # how many windows triggered
            "total_windows":  int,
            "class_idx":      int,
        }
        """
        waveform = self._load_waveform(audio_path)
        total_samples = waveform.shape[1]
        chunk_samples = SAMPLE_RATE * MAX_DURATION
        hop_samples   = chunk_samples // 2          # 50% overlap

        # Build list of (start, end) sample positions
        starts = list(range(0, max(1, total_samples - chunk_samples + 1), hop_samples))
        if not starts:
            starts = [0]

        best_positive_conf = 0.0
        positive_windows   = 0

        with torch.no_grad():
            for start in starts:
                chunk = waveform[:, start : start + chunk_samples]
                tensor = self._waveform_to_mel(chunk)
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                class_idx  = int(probs.argmax())
                confidence = float(probs[class_idx])

                if class_idx == POSITIVE_IDX:
                    positive_windows += 1
                    if confidence > best_positive_conf:
                        best_positive_conf = confidence

        is_positive = positive_windows > 0

        return {
            "is_positive":      is_positive,
            "confidence":       best_positive_conf if is_positive else float(probs[POSITIVE_IDX]),
            "positive_windows": positive_windows,
            "total_windows":    len(starts),
            "class_idx":        POSITIVE_IDX if is_positive else (1 - POSITIVE_IDX),
        }


# ------------------------------------------------------------------ #
#  Singleton                                                           #
# ------------------------------------------------------------------ #

_instance: SoundClassifier | None = None


def get_classifier(model_path: str | None = None) -> SoundClassifier:
    global _instance
    if _instance is None:
        path = model_path or os.getenv("MODEL_PATH", "models/model.pth")
        _instance = SoundClassifier(path)
    return _instance