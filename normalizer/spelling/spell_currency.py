import io

from normalizer.utils.helpers import CURRENCY_SYMBOLS
from normalizer.spelling.spell_integer import spell_integer
from normalizer.spelling.spell_float import spell_float 

def spell_currency(symbol, amount):
    if not isinstance(amount, (int, float)):
        raise ValueError('Üzr istəyirik! Məbləğ tipində olmalıdır.')
        
    elif isinstance(amount, float):
        amount_in_words = spell_float(float(amount))
    else:
        amount_in_words = spell_integer(int(amount))

    # replace currency with amount in words
    new_symbol = CURRENCY_SYMBOLS.get(symbol, "")

    return amount_in_words + " " + new_symbol