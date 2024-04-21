import random
import numpy as np
import torch
import torch.utils.data
import tiktoken

import models.layers as layers
from utils import load_wav_to_torch, load_filepaths_and_text
from hparams import create_hparams
from texts import text_to_sequence
from torchaudio.transforms import MelSpectrogram, Spectrogram
import torchaudio.functional as F
import librosa
from datasets.mel_processing import spectrogram_torch

tokenizer = tiktoken.get_encoding("cl100k_base")


hparams = create_hparams()

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams = hparams, mode = "train"):
        self.audiopaths_and_text = load_filepaths_and_text(hparams)
        self.mode = mode
        self.audiopaths_and_text = self.split_val_train(self.audiopaths_and_text, self.mode)
        
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
        
        self.melspectrogram = MelSpectrogram(sample_rate = self.stft.sampling_rate, n_fft = hparams["filter_length"], win_length = hparams['win_length'],
                                             hop_length = hparams['hop_length'], f_min = hparams["mel_fmin"], f_max = hparams["mel_fmax"], n_mels = hparams['n_mel_channels'])
        
        #melspec = librosa.feature.melspectrogram(y= audio_norm.detach().numpy(), sr=self.stft.sampling_rate, n_fft=hparams["filter_length"], hop_length=hparams['hop_length'],
        #                                         win_length=hparams['win_length'], n_mels = hparams['n_mel_channels'], fmin = hparams["mel_fmin"], 
        #                                         fmax = hparams["mel_fmax"])

    def get_mel_text_pair(self, index):
        # separate filename and text
        audiopath, text = self.audiopaths_and_text[0][index], self.audiopaths_and_text[1][index]
        #text = "salam, mənim 25 pişiyim var. hərəsini $3.52 almışam. Kamran bilsə məni 25 dəfə öldürər"
        text = torch.Tensor(self.vectorizer(text.lower(), ["english_cleaners"])).long()
        mel = self.get_mel(audiopath.replace("book_sounds/original_book/", "datasets/voice_book/book_sounds/"))
        return (text.to(hparams["device"]), mel)
    
    
    def split_val_train(self, paths, mode = "train"):
        audio, transcription = paths
        train_audio = []
        train_tr = []
        val_audio = []
        val_tr = []
        for i in range(len(audio)):
            if (i%20!=0):
                train_audio.append(audio[i])
                train_tr.append(transcription[i])
            else:
                val_audio.append(audio[i])
                val_tr.append(transcription[i])

        if(mode =="train"):
            return train_audio, train_tr
        else:
            return val_audio, val_tr
            
    def get_mel(self, filename):

        audio, sampling_rate = load_wav_to_torch(filename)
        
        
        if sampling_rate != self.stft.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate)
        audio = self.process_audio(audio)
        audio_norm = audio / self.max_wav_value
        #audio_norm = audio_norm.unsqueeze(0)
        #audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        
        melspec = librosa.feature.melspectrogram(y= audio_norm.detach().numpy(), sr=self.stft.sampling_rate, n_fft=hparams["filter_length"], hop_length=hparams['hop_length'],
                                                 win_length=hparams['win_length'], n_mels = hparams['n_mel_channels'], fmin = hparams["mel_fmin"], 
                                                 fmax = hparams["mel_fmax"])
        melspec = torch.Tensor(melspec)

        return melspec
    
    
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
    
    def tokenize_text(self, text):
        text = self.tokenizer.encode(text)
        return torch.IntTensor(text)

    def get_text(self, text):
        
        text_norm = torch.IntTensor(-(text, self.text_cleaners))
        #text_norm = []
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(index)

    def __len__(self):
        return len(self.audiopaths_and_text[0])


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per 
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            
        

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        #print(max_target_len)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        
        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len).zero_()
        #print(len(batch))
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            

            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
        
        return text_padded.to(hparams["device"]), input_lengths, mel_padded, gate_padded, \
            output_lengths
            
            
            


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams = hparams, mode = "train"):
        self.audiopaths_and_text = load_filepaths_and_text(hparams)
        self.mode = mode
        self.audiopaths_and_text = self.split_val_train(self.audiopaths_and_text, self.mode)
        
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
        audiopath, text = self.audiopaths_and_text[0][index], self.audiopaths_and_text[1][index]
        #text = "salam, mənim 25 pişiyim var. hərəsini $3.52 almışam. Kamran bilsə məni 25 dəfə öldürər"
        text = torch.Tensor(self.vectorizer(text, ["english_cleaners"]))
        mel, audio = self.get_mel(audiopath.replace("book_sounds/original_book/", "datasets/voice_book/book_sounds/"))
        return text, audio
    
    
    def split_val_train(self, paths, mode = "train"):
        audio, transcription = paths
        train_audio = []
        train_tr = []
        val_audio = []
        val_tr = []
        for i in range(len(audio)):
            if (i%20!=0):
                train_audio.append(audio[i])
                train_tr.append(transcription[i])
            else:
                val_audio.append(audio[i])
                val_tr.append(transcription[i])

        if(mode =="train"):
            return train_audio, train_tr
        else:
            return val_audio, val_tr
            
    def get_mel(self, filename):

        audio, sampling_rate = load_wav_to_torch(filename)
        
        
        if sampling_rate != self.stft.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate)
        audio = self.process_audio(audio)
        audio_norm = audio / self.max_wav_value
        #audio_norm = audio_norm.unsqueeze(0)
        #audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        
        melspec = librosa.feature.melspectrogram(y= audio_norm.detach().numpy(), sr=self.stft.sampling_rate, n_fft=hparams["filter_length"], hop_length=hparams['hop_length'],
                                                 win_length=hparams['win_length'], n_mels = hparams['n_mel_channels'], fmin = hparams["mel_fmin"], 
                                                 fmax = hparams["mel_fmax"])
        melspec = torch.Tensor(melspec)
        
        #melspec = torch.stft(audio_norm, n_fft=hparams["filter_length"], hop_length=hparams["hop_length"], win_length=hparams["win_length"],return_complex=True)#self.stft.mel_spectrogram(audio_norm)
        #melspec = librosa_mel_fn(sr = sampling_rate, n_fft = hparams["filter_length"], n_mels= hparams["n_mel_channels"], fmin = hparams["mel_fmin"], fmax = hparams["mel_fmax"])
        # melspec = torch.from_numpy(melspec).float()
        #melspec = torch.squeeze(melspec, 0)
        #print(melspec.shape)
        
        # audio_norm = audio_norm.unsqueeze(0)
        # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # melspec = self.stft.mel_spectrogram(audio_norm)
        # melspec = torch.squeeze(melspec, 0)

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
    
    def tokenize_text(self, text):
        text = self.tokenizer.encode(text)
        return torch.IntTensor(text)

    def get_text(self, text):
        
        text_norm = torch.IntTensor(-(text, self.text_cleaners))
        #text_norm = []
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(index)

    def __len__(self):
        return len(self.audiopaths_and_text[0])


class TextMelAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams = hparams, mode = "train"):
        self.audiopaths_and_text = load_filepaths_and_text(hparams)
        self.mode = mode
        self.audiopaths_and_text = self.split_val_train(self.audiopaths_and_text, self.mode)
        
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
        self.melspectrogram = MelSpectrogram(sample_rate = self.stft.sampling_rate, n_fft = hparams["filter_length"], win_length = hparams['win_length'],
                                             hop_length = hparams['hop_length'], f_min = hparams["mel_fmin"], f_max = hparams["mel_fmax"], n_mels = hparams['n_mel_channels'])
        self.spectrogram = Spectrogram(n_fft = hparams["filter_length"], win_length = hparams['win_length'],
                                             hop_length = hparams['hop_length'] )

    def get_mel_text_pair(self, index):
        # separate filename and text
        audiopath, text = self.audiopaths_and_text[0][index], self.audiopaths_and_text[1][index]
        #text = "salam, mənim 25 pişiyim var. hərəsini $3.52 almışam. Kamran bilsə məni 25 dəfə öldürər"
        text = torch.Tensor(self.vectorizer(text, ["english_cleaners"]))
        mel, audio = self.get_mel(audiopath.replace("book_sounds/original_book/", "datasets/voice_book/book_sounds/"))
        return text, mel, audio
    
    
    def split_val_train(self, paths, mode = "train"):
        audio, transcription = paths
        train_audio = []
        train_tr = []
        val_audio = []
        val_tr = []
        for i in range(len(audio)):
            if (i%20!=0):
                train_audio.append(audio[i])
                train_tr.append(transcription[i])
            else:
                val_audio.append(audio[i])
                val_tr.append(transcription[i])

        if(mode =="train"):
            return train_audio, train_tr
        else:
            return val_audio, val_tr
            
    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate)
        audio = self.process_audio(audio)
        
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        spec = spectrogram_torch(audio_norm, hparams["filter_length"],
            self.sampling_rate, hparams['hop_length'], hparams['win_length'],
            center=False)
        
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm        
    
    def get_mel(self, filename):

        audio, sampling_rate = load_wav_to_torch(filename)
        
        
        if sampling_rate != self.stft.sampling_rate:
            audio = self.resample_audio(audio, sampling_rate)
        #audio = self.process_audio(audio)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        #audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        
        #melspec = librosa.feature.melspectrogram(y= audio_norm.detach().numpy(), sr=self.stft.sampling_rate, n_fft=hparams["filter_length"], hop_length=hparams['hop_length'],
        #                                         win_length=hparams['win_length'], n_mels = hparams['n_mel_channels'], fmin = hparams["mel_fmin"], 
        #                                         fmax = hparams["mel_fmax"])
        #melspec = torch.Tensor(melspec)
        #print("inside dataloader". melspec.shape)
        melspec = self.melspectrogram(audio_norm)

        return melspec.squeeze(0), audio_norm
    
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
    
    def tokenize_text(self, text):
        text = self.tokenizer.encode(text)
        return torch.IntTensor(text)

    def get_text(self, text):
        
        text_norm = torch.IntTensor(-(text, self.text_cleaners))
        #text_norm = []
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(index)

    def __len__(self):
        return len(self.audiopaths_and_text[0])
    
    
    
    
class TextAudioCollate():
    """ Zero-pads model inputs and targets based on number of frames per 
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        max_ln = max([len(batch[i][0]) for i in range(len(batch))])
        new_text_batch = []
        new_audio_batch = []
        for item in range(len(batch)):
            
            if(len(batch[item][0])< max_ln):
                pad = torch.tensor([0  for j in range(len(batch[item][0]), max_ln)])
            
                new_text_batch.append( torch.cat((batch[item][0], pad ), dim=0))
            
            else:
                new_text_batch.append( batch[item][0])
            new_audio_batch.append(batch[item][1])
                
            #print(len(batch[item][0]))
            
            # Right zero-pad all one-hot text sequences to max input length
        return new_text_batch, new_audio_batch
        
            
        

class TextAudioCollate_Vits():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2][0].size(0) for x in batch])
        

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
        # if self.return_ids:
        #     return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths
        # return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths
        if self.return_ids:
            return text_padded, text_lengths, spec_padded, wav_padded, wav_lengths
        return text_padded, text_lengths, spec_padded, wav_padded, wav_lengths
            
            

class TextMelAudioCollate():
    """ Zero-pads model inputs and targets based on number of frames per 
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        
        
        
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            
        

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        #print(max_target_len)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        
        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len).zero_()
        #print(len(batch))
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            

            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            
            
        
        
        max_wav_len = max([x[2][0].size(0) for x in batch])
        wav_lengths = torch.LongTensor(len(batch))
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            
            row = batch[ids_sorted_decreasing[i]]
            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)


        
        
        return text_padded.to(hparams["device"]), input_lengths, mel_padded, gate_padded, \
            output_lengths, wav_padded