import torch 
import torch.nn as nn

from loss import WaveGlowLoss, Tacotron2Loss


class trainer(nn.Module):
    def __init__(self, tacotron = None, waveglow = None, hparams = None):
        super(trainer, self).__init__()
        self.tacotron = tacotron
        self.waveglow = waveglow
        self.hparams = hparams
        self.optimizer = torch.optim.SGD(list(self.tacotron.parameters()) + list(self.waveglow.parameters()), lr=self.hparams["learning_rate_wv"])
        
        self.loss_tacotron = Tacotron2Loss()
        self.loss_waveglow = WaveGlowLoss(hparams["sigma"])

    def train_step(self, x):            
        self.optimizer.zero_grad()
        self.tacotron.zero_grad()
        self.waveglow.zero_grad()
        text, mel_gt = self.tacotron.parse_batch(x[:-1])
        y_pred = self.tacotron(text)
        
        loss_tacotron = self.loss_tacotron(y_pred, mel_gt)
        
        audio = torch.autograd.Variable(x[-1].cuda()).squeeze(1)
        
        outputs = self.waveglow((y_pred[0].cuda(), audio)) 
        loss_waveglow = self.loss_waveglow(outputs)
        loss = loss_tacotron + loss_waveglow
        
        return {"loss": loss, "loss tacotron": loss_tacotron.item(), "loss waveglow": loss_waveglow.item()}
    
    def forward(self, x):
        
        self.tacotron.train()
        self.waveglow.train()
        
        
        losses = self.train_step(x)
        losses["loss"].backward()
        
        value = {"loss tacotron": losses["loss tacotron"], "loss waveglow": losses["loss waveglow"]}
        
        del losses
        self.optimizer.step()
        
        
        return value
    
    
    def infer(self, x):
        self.tacotron.eval()
        self.waveglow.eval()
        
        with torch.no_grad():

            x, _, y = x
            mel_pred = self.tacotron.inference(x)[0]

            audio = torch.autograd.Variable(y.cuda())
            outputs = self.waveglow((mel_pred, audio))
            loss = self.loss_waveglow(outputs)
            reduced_val_loss = loss.item()
        
        return reduced_val_loss
            
        
        