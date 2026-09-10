"""
model.py — PyTorch inference wrapper
Loads your trained .pth model and runs classification on audio files.
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
SAMPLE_RATE    = int(os.getenv("MODEL_SAMPLE_RATE",          "12000"))
N_MELS         = int(os.getenv("MODEL_N_MELS",               "64"))
N_FFT          = int(os.getenv("MODEL_N_FFT",                 "512"))
HOP_LENGTH     = int(os.getenv("MODEL_HOP_LENGTH",            "128"))
CLIP_DURATION  = int(os.getenv("MODEL_CLIP_DURATION_SEC",     "3"))    # ← changed: 3-s chunks
OVERLAP        = float(os.getenv("MODEL_OVERLAP",             "0.5"))  # ← new: 50% overlap
POSITIVE_IDX   = int(os.getenv("MODEL_POSITIVE_CLASS_IDX",   "0"))
LABEL_NAMES    = {0: "chainsaw", 1: "environment"}


class SoundClassifier:
    """
    Thin wrapper around any torchscript / state-dict model.

    Expected model behaviour
    ------------------------
    Input  : (B, 1, N_MELS, T) float tensor (normalised log-mel spectrogram)
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

        # Inject architecture classes so pickle can find them
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
        # ← changed: use torchaudio (matches friend's code)
        import soundfile as sf

        data, sr = sf.read(audio_path)

        waveform = torch.tensor(data).float()

        # Stereo → mono
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T

    # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        return waveform  # (1, N)  # (1, N)

    def _make_chunks(self, waveform: torch.Tensor) -> list:
        """
        Slide a window of CLIP_DURATION seconds over the waveform with OVERLAP.
        Last chunk is zero-padded if too short.
        Returns a list of (1, chunk_samples) tensors.
        """
        chunk_samples = SAMPLE_RATE * CLIP_DURATION
        hop_samples   = int(chunk_samples * (1 - OVERLAP))
        total         = waveform.shape[-1]
        chunks        = []

        start = 0
        while start < total:
            clip = waveform[:, start : start + chunk_samples]
            if clip.shape[-1] < chunk_samples:
                clip = nn.functional.pad(clip, (0, chunk_samples - clip.shape[-1]))
            chunks.append(clip)
            start += hop_samples
            if start >= total:
                break

        return chunks

    def _chunk_to_mel(self, chunk: torch.Tensor) -> torch.Tensor:
        """(1, chunk_samples) → (1, 1, N_MELS, T_frames) normalised log-mel tensor."""
        chunk = chunk.to(self.device)
        mel    = self._mel(chunk)           # (1, N_MELS, T_frames)
        mel_db = self._amp_to_db(mel)
        # ← changed: apply min-max normalisation (matches friend's code)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return mel_db.unsqueeze(0)          # (1, 1, N_MELS, T_frames)

    def predict(self, audio_path: str) -> dict:
        """
        Sliding-window inference with soft voting.

        The clip is split into CLIP_DURATION-second windows with OVERLAP.
        Softmax probabilities are averaged across all windows (soft voting),
        and the argmax of the average is the final prediction.

        Returns
        -------
        {
            "is_positive":      bool,
            "confidence":       float,   # confidence of the winning class
            "positive_windows": int,     # how many windows individually predicted positive
            "total_windows":    int,
            "class_idx":        int,
            "label":            str,
        }
        """
        waveform = self._load_waveform(audio_path)
        chunks   = self._make_chunks(waveform)

        # Batch all chunks in one forward pass
        mel_batch = torch.cat([self._chunk_to_mel(c) for c in chunks], dim=0)
        # mel_batch: (num_chunks, 1, N_MELS, T_frames)

        with torch.no_grad():
            logits = self.model(mel_batch)                      # (num_chunks, 2)
            probs  = torch.softmax(logits, dim=1).cpu()         # (num_chunks, 2)

        # ← changed: soft voting — average probs across all chunks
        avg_probs  = probs.mean(dim=0)                          # (2,)
        class_idx  = int(avg_probs.argmax().item())
        confidence = float(avg_probs[class_idx].item())

        # Count how many individual windows predicted positive
        positive_windows = int((probs.argmax(dim=1) == POSITIVE_IDX).sum().item())
        is_positive      = (class_idx == POSITIVE_IDX)

        return {
            "is_positive":      is_positive,
            "confidence":       confidence,
            "positive_windows": positive_windows,
            "total_windows":    len(chunks),
            "class_idx":        class_idx,
            "label":            LABEL_NAMES[class_idx],
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