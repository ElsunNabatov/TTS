# import sys
# sys.path.append('C:/Users/Asmar/OneDrive/Masaüstü/PRODATA/normalizer')

from normalizer.spelling.spell_integer import spell_integer
from normalizer.spelling.spell_float import spell_float
from normalizer.spelling.spell_currency import spell_currency
from normalizer.spelling.spell_date import spell_date

from normalizer.utils.helpers import find_numbers_and_currencies_in_text, find_dates_in_text

def is_integer(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

def transform(text):
    dates = find_dates_in_text(text)
    for (day, month, year) in dates:
        chunk = spell_date(day, month, year)
        text = text.replace(f"{day}/{month}/{year}", chunk)
        text = text.replace(f"{day}.{month}.{year}", chunk)
        text = text.replace(f"{day}-{month}-{year}", chunk)

    items = find_numbers_and_currencies_in_text(text)
    for (symbol, amount) in items:
        if symbol is None:  # there is no currency symbol
            if is_integer(amount):  # the number is an integer
                chunk = spell_integer(amount)
            else:  # the number is a float
                chunk = spell_float(amount)

            text = text.replace(str(amount), chunk)

        else:  # there is currency symbol
            chunk = spell_currency(symbol, amount)

            text = text.replace(f"{symbol} {amount}0", chunk)
            text = text.replace(f"{symbol} {amount}", chunk)
            text = text.replace(f"{symbol}{amount}0", chunk)
            text = text.replace(f"{symbol}{amount}", chunk)
            text = text.replace(f"{amount}0 {symbol}", chunk)
            text = text.replace(f"{amount} {symbol}", chunk)
            text = text.replace(f"{amount}0{symbol}", chunk)
            text = text.replace(f"{amount}{symbol}", chunk)

    return text


