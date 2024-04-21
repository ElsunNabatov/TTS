import torch
from loss import WaveGlowLoss
from models.waveglow import WaveGlow
from datasets.audio_dataloader import TextMelLoader, TextMelCollate, TextMelAudioLoader
from datasets.unlabelled_audio import MelAudioLoader
from torch.utils.data import DataLoader
import tqdm
from hparams import create_hparams
import json
from torch.utils.tensorboard import SummaryWriter


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = MelAudioLoader(hparams)
    valset = MelAudioLoader(hparams, mode = "val")
    shuffle = True

    train_loader = DataLoader(trainset, shuffle=shuffle,
                              batch_size=hparams["batch_size"],
                              drop_last=True, num_workers=4)
    print("train", len(train_loader))
    
    val_loader = DataLoader(valset, num_workers=4,
                                shuffle=False, batch_size=1)
    return train_loader, val_loader

def load_model(hparams):
    model = WaveGlow() #TODO import configuraiton file

    return model.to(hparams["device"])


def validate(model, criterion, val_loader, epoch, writer, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
    
    with torch.no_grad():

        tq = tqdm.tqdm(total=len(val_loader))
        tq.set_description("Validation")
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = batch
            mel = torch.autograd.Variable(x.cuda())
            audio = torch.autograd.Variable(y.cuda())
            outputs = model((mel, audio))
            loss = criterion(outputs)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            tq.update(1)
        
        tq.close()
        val_loss = val_loss / (i + 1)
    model.train()
    print("Validation loss {}: {:9f}  ".format(epoch, val_loss))
    writer.add_scalar("Loss_validation", val_loss, epoch)
    #logger.log_validation(val_loss, model, y, y_pred, iteration)



def train_waveglow():
    writer = SummaryWriter(comment="_waveglow" )
    hparams = create_hparams()
    criterion = WaveGlowLoss(hparams["sigma"])
    model = WaveGlow(hparams).cuda()



    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate_wv"])

    train_loader, val_loader = prepare_dataloaders(hparams)


    # Load checkpoint if one exists


    curr_epoch = 0
    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(curr_epoch, hparams["epochs"]):
        total_loss = 0
        
        print("Epoch: {}".format(epoch))
        total_loss = 0
        validate(model, criterion, val_loader, epoch, writer, hparams)
        tq = tqdm.tqdm(total=len(train_loader)) 
        tq.set_description('epoch %d' % (epoch))
        for i, batch in enumerate(train_loader):
            
            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model((mel, audio))

            loss = criterion(outputs)

            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
        tq.close()
        writer.add_scalar("Loss", total_loss/len(train_loader), epoch)
        

train_waveglow()