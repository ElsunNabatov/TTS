import torch
from loss import WaveGlowLoss
from models.waveglow import WaveGlow
from datasets.audio_dataloader import TextMelAudioLoader, TextMelAudioCollate
from datasets.unlabelled_audio import MelAudioLoader
from torch.utils.data import DataLoader
import tqdm
from hparams import create_hparams
import json
from utils import load_checkpoint

from models.model import Tacotron2
from torch.utils.tensorboard import SummaryWriter
from trainer_tacotron import trainer



def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelAudioLoader(hparams)
    valset = TextMelAudioLoader(hparams, mode = "val")
    collate_fn = TextMelAudioCollate(hparams["n_frames_per_step"])

    shuffle = True

    train_loader = DataLoader(trainset, shuffle=shuffle,
                              batch_size=hparams["batch_size"],
                              drop_last=True, collate_fn=collate_fn)
    #print("train", len(train_loader))
    
    val_loader = DataLoader(valset,
                                shuffle=False, batch_size=1,
                                collate_fn=collate_fn)
    return train_loader, val_loader



def load_model(hparams):
    tacotron = Tacotron2(hparams)
    waveglow = WaveGlow(hparams) #DONE import configuraiton file
    
    #if(hparams["device"]=="cuda"):
    #    model = load_checkpoint(model)
    
    return tacotron.to(hparams["device"]), waveglow.to(hparams["device"])



def validate(trainer, criterion, val_loader, epoch, writer, hparams):
    """Handles all the validation scoring and printing"""

    tq = tqdm.tqdm(total=len(val_loader))
    tq.set_description("Validation")
    val_loss = 0.0
    
    for i, batch in enumerate(val_loader):
        reduced_val_loss = trainer.infer(batch)
        val_loss += reduced_val_loss
        tq.update(1)
    
    tq.close()
    val_loss = val_loss / (i + 1)
    print("Validation loss {}: {:9f}  ".format(epoch, val_loss))
    writer.add_scalar("Loss_validation", val_loss, epoch)
    #logger.log_validation(val_loss, model, y, y_pred, iteration)



def train_waveglow():
    writer = SummaryWriter(comment="_offline_Tacotron" )
    hparams = create_hparams()

    train_loader, val_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    tacotron, waveglow = load_model(hparams)
    trainer_main = trainer(tacotron, waveglow, hparams)
    curr_epoch = 0

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(curr_epoch, hparams["epochs"]):
        total_loss_t = 0
        total_loss_w = 0
        
        
        #validate(model, criterion, val_loader, epoch, writer, hparams)
        tq = tqdm.tqdm(total=len(train_loader)) 
        tq.set_description('epoch %d' % (epoch))
        
        for i, batch in enumerate(train_loader):
            
            loss = trainer_main(batch)
            total_loss_t = total_loss_t + loss['loss tacotron']
            total_loss_w = total_loss_w + loss['loss waveglow']
            
            tq.set_postfix(loss_tacotron='%.6f, loss_waveglow = %.6f' % (loss['loss tacotron'], loss['loss waveglow']))
            tq.update(1)
        
        tq.close()
        writer.add_scalar("Loss for waveglow", total_loss_w/len(train_loader), epoch)
        writer.add_scalar("Loss for Tacotron", total_loss_t/len(train_loader), epoch)
        

train_waveglow()

# hparams = create_hparams()
# dt, _ = prepare_dataloaders(hparams)
# it = enumerate(dt)
# _, batch = next(it)

# print(len(batch[0][0]), len(batch[0][1]), len(batch[0][2]))