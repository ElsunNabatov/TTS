import os
import time
import argparse
import math
from numpy import finfo

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import Tacotron2
from datasets.audio_dataloader import TextMelLoader, TextMelCollate
from loss import Tacotron2Loss
import tqdm
from hparams import create_hparams
from utils import load_checkpoint, save_checkpoint



def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams)
    valset = TextMelLoader(hparams, mode = "val")
    collate_fn = TextMelCollate(hparams["n_frames_per_step"])

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
    model = Tacotron2(hparams)
    
    # if(hparams["device"]=="cuda"):
    #     model = load_checkpoint(model)
    
    return model.to(hparams["device"])


def validate(model, criterion, val_loader, epoch, writer, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
    collate_fn = TextMelCollate(hparams["n_frames_per_step"])
    
    with torch.no_grad():

        tq = tqdm.tqdm(total=len(val_loader))
        tq.set_description("Validation")
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            tq.update(1)
        
        tq.close()
        val_loss = val_loss / (i + 1)
    model.train()
    print("Validation loss {}: {:9f}  ".format(epoch, val_loss))
    writer.add_scalar("Loss_validation", val_loss, epoch)
    #logger.log_validation(val_loss, model, y, y_pred, iteration)

def train():
    writer = SummaryWriter(comment="_tacotron" )
    hparams = create_hparams()
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """


    #torch.manual_seed(hparams["seed"])
    #torch.cuda.manual_seed(hparams["seed"])

    model = load_model(hparams)
    learning_rate = hparams["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams["weight_decay"])


    criterion = Tacotron2Loss()
    


    train_loader, val_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    
    total_loss = 0
    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams["epochs"]):
        total_loss = 0

        tq = tqdm.tqdm(total=len(train_loader)) 
        tq.set_description('epoch %d' % (epoch))
        for i, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)


            total_loss +=loss.item()

            loss.backward()


            optimizer.step()

                    
            iteration += 1
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
        tq.close()
        writer.add_scalar("Loss", total_loss/len(train_loader), epoch)
        #writer.add_scalar("mIoU", miou, epoch)
        validate(model, criterion, val_loader, epoch, writer, hparams)
        save_checkpoint(model, "aze_test.pth")


train()
