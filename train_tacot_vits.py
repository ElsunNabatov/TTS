import torch
import librosa
from torch import nn, optim
from torchaudio.transforms import MelSpectrogram
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from hparams import create_hparams
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from datasets.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import utils
from datasets.audio_dataloader import (
  TextMelAudioLoader,
  TextAudioCollate_Vits,
)

from utils import load_checkpoint
from models.vits import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from loss import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from texts.symbols import symbols
import tqdm

