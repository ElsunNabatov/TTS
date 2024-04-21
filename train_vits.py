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


import utils
hps = utils.get_hparams_from_file("configs/ljs_base.json")


hparams = create_hparams()



def get_models(hps, hparams):
    
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(hparams["device"])
    
    net_g = load_checkpoint(net_g, path_to_checkpoint="checkpoints/G_100000.pth")
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(hparams["device"])

    return net_g, net_d

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelAudioLoader(hparams)
    valset = TextMelAudioLoader(hparams, mode = "val")
    collate_fn = TextAudioCollate_Vits(hparams["n_frames_per_step"])

    shuffle = True

    train_loader = DataLoader(trainset, shuffle=False,
                              batch_size=hparams["batch_size"],
                              drop_last=True, collate_fn=collate_fn)
    #print("train", len(train_loader))
    
    val_loader = DataLoader(valset,
                                shuffle=False, batch_size=1,
                                collate_fn=collate_fn)
    return train_loader, val_loader
    


def train( hps, hparams):
    scaler = GradScaler(enabled=hps.train.fp16_run)
    writer = SummaryWriter(comment="_vits_fine")
    
  
    net_g, _ = get_models(hps, hparams)
    
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    

    
    
    train_loader, val_loader = prepare_dataloaders(hparams)


    net_g.train()
    transform = MelSpectrogram(sample_rate=hparams['sampling_rate'], n_fft=hparams["filter_length"], hop_length=hparams['hop_length'],
                                                 win_length=hparams['win_length'], n_mels = hparams['n_mel_channels'], f_min = hparams["mel_fmin"], 
                                                 f_max = hparams["mel_fmax"])
    
    max_epochs = hparams["epochs"]
    for epoch in range(max_epochs):
        loss_d_total = 0
        loss_kl_total = 0
        
        tq = tqdm.tqdm(total=len(train_loader)) 
        tq.set_description('epoch %d' % (epoch))
      
        for _, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
            
            x, x_lengths = x.to(hparams["device"]), x_lengths.to(hparams["device"])
            
            spec, spec_lengths = spec.to(hparams["device"]), spec_lengths.to(hparams["device"])
            
            y, y_lengths = y.to(hparams["device"]), y_lengths.to(hparams["device"])


            with autocast(enabled=hps.train.fp16_run):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)
                
                mel = spec_to_mel_torch(
                    spec, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax)
                y_mel = mel



                with autocast(enabled=hps.train.fp16_run):
                    # Generator
                    with autocast(enabled=False):
                        loss_dur = torch.sum(l_length.float())
                        #loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                        loss_gen_all = loss_dur + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = utils.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()
            tq.set_postfix(loss_kl='%.6f, loss_dur = %.6f' % (loss_kl.item(), loss_dur.item()))
            tq.update(1)
            lr = optim_g.param_groups[0]['lr']
            loss_d_total += loss_dur.item()
            loss_kl_total += loss_kl.item()
        writer.add_scalar("Loss duration", loss_d_total/len(train_loader), epoch)
        writer.add_scalar("Loss KL", loss_kl_total/len(train_loader), epoch)
        tq.close()
            
train(hps, hparams)
            
        
                


    


 