import random
import numpy as np
import torch
import torch.utils.data
import tiktoken

import models.layers as layers
from utils import load_wav_to_torch, load_filepaths_and_text, load_filepaths
from hparams import create_hparams
from texts import text_to_sequence
import torchaudio.functional as F
import librosa

hparams = create_hparams()


class MelAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams = hparams, mode = "train"):
        self.audiopaths_and_text = load_filepaths()
        self.mode = mode
        self.audiopaths_and_text = self.split_val_train(self.audiopaths_and_text, self.mode)
        #print(len(self.audiopaths_and_text))
        self.segment_length = hparams["segment_length"]
        self.text_cleaners = hparams["text_cleaners"]
        self.max_wav_value = hparams["max_wav_value"]
        self.sampling_rate = hparams["sampling_rate"]
        self.stft = layers.TacotronSTFT(
            hparams["filter_length"], hparams["hop_length"], hparams["win_length"],
            hparams["n_mel_channels"], hparams["sampling_rate"], hparams["mel_fmin"],
            hparams["mel_fmax"])
        random.seed(hparams["seed"])
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vectorizer = text_to_sequence


    def get_mel_text_pair(self, index):
        # separate filename and text
        audiopath = self.audiopaths_and_text[index]
        #text = "salam, mənim 25 pişiyim var. hərəsini $3.52 almışam. Kamran bilsə məni 25 dəfə öldürər"
        #text = torch.Tensor(self.vectorizer(text, ["english_cleaners"]))
        mel, audio = self.get_mel(audiopath.replace("book_sounds/original_book/", "datasets/voice_book/book_sounds/"))
        return (mel, audio)
    
    
    def split_val_train(self, paths, mode = "train"):
        train_audio = []
        val_audio = []
        for i in range(len(paths)):
            if (i%20!=0):
                train_audio.append(paths[i])
                #train_tr.append(transcription[i])
            else:
                val_audio.append(paths[i])
                #val_tr.append(transcription[i])

        if(mode =="train"):
            return train_audio
        else:
            return val_audio
            
    def get_mel(self, filename):

        audio, sampling_rate = load_wav_to_torch(filename)
        
        
        if sampling_rate != self.stft.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate)
        audio = self.process_audio(audio)
        audio_norm = audio / self.max_wav_value

        
        melspec = librosa.feature.melspectrogram(y= audio_norm.detach().numpy(), sr=self.stft.sampling_rate, n_fft=hparams["filter_length"], hop_length=hparams['hop_length'],
                                                 win_length=hparams['win_length'], n_mels = hparams['n_mel_channels'], fmin = hparams["mel_fmin"], 
                                                 fmax = hparams["mel_fmax"])
        melspec = torch.Tensor(melspec)

        return melspec, audio_norm
    
    def resample_audio(self, audio, sampling_rate):
        mono_audio = torch.mean(audio, dim=0)
        resampled_waveform = F.resample(mono_audio, sampling_rate, self.sampling_rate, resampling_method="sinc_interp_kaiser")
        return resampled_waveform
    
    def process_audio(self, audio):
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
        return audio
    

    def __getitem__(self, index):
        index =1
        return self.get_mel_text_pair(index)

    def __len__(self):
        return len(self.audiopaths_and_text)
