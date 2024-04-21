""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from texts import cmudict
from hparams import create_hparams
from utils import TextMapper


'''

This part corresponds tacotron preprocessing.

we need use the same preprocessing that meta have used for Azerbaijani
That is why we divide our script into 2 parts

'''

hparams = create_hparams()



if (hparams["training_model"] == "tacotron" ):
    _pad        = '_'
    _punctuation = '!\'(),.:;? '
    _special = '-'
    _letters = 'ABCÇDEƏFGĞHXİIJKQLMNOÖPRSŞTUÜVYZabcçdeəfgğhxiıjkqlmnoöprsştuüvyz'
    #_letters = 'abcçdeəfgğhxiıjkqlmnoöprsştuüvyz'
    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = ['@' + s for s in cmudict.valid_symbols]

    # Export all symbols:
    symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
else:
    mapper = TextMapper("configs/vocab.txt")
    symbols = mapper.symbols
    
    
    
    

