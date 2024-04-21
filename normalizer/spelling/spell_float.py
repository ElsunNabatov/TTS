from normalizer.spelling.spell_integer import spell_integer

from normalizer.utils.translations import HUNDREDS_PREFIX, HUNDREDS_AS_WORDS, POINT_AS_WORD
from normalizer.utils.helpers import THOUSAND, MILLION, BILLION, TRILLION, QUADRILLION

def spell_float(number):
    number_chunks = str(number).split('.')
    integer_part = int(number_chunks[0])
    floating_part = int(number_chunks[1])
    floating_part_length = len(number_chunks[1])

    numOfFr = ''
    exponent = 1

    if floating_part_length == 2:
        exponent = 2
    elif floating_part_length >= 3 and floating_part_length < 6:
        exponent = 3
        numOfThousands = floating_part // THOUSAND
        if 1 < numOfThousands < 10:
            numOfThousands = 10
        elif 10 < numOfThousands < 100:
            numOfThousands = 100
        numOfFr = f"{spell_integer(numOfThousands, False)} " if numOfThousands > 1 else ""
    elif floating_part_length >= 6 and floating_part_length < 9:
        exponent = 6
        numOfMillions = floating_part // MILLION
        if 1 < numOfMillions < 10:
            numOfMillions = 10
        elif 10 < numOfMillions < 100:
            numOfMillions = 100
        numOfFr = f"{spell_integer(numOfMillions, False)} " if numOfMillions > 1 else ""
    elif floating_part_length >= 9 and floating_part_length < 12:
        exponent = 9
        numOfBillions = floating_part // BILLION
        if 1 < numOfBillions < 10:
            numOfBillions = 10
        elif 10 < numOfBillions < 100:
            numOfBillions = 100
        numOfFr = f"{spell_integer(numOfBillions, False)} " if numOfBillions > 1 else ""
    elif floating_part_length >= 12 and floating_part_length < 15:
        exponent = 12
        numOfTrillions = floating_part // TRILLION
        if 1 < numOfTrillions < 10:
            numOfTrillions = 10
        elif 10 < numOfTrillions < 100:
            numOfTrillions = 100
        numOfFr = f"{spell_integer(numOfTrillions, False)} " if numOfTrillions > 1 else ""
    elif 15 <= floating_part_length <= 16:
        exponent = 15
        numOfQuadrillions = floating_part // QUADRILLION
        if 1 < numOfQuadrillions < 10:
            numOfQuadrillions = 10
        elif 10 < numOfQuadrillions < 100:
            numOfQuadrillions = 100
        numOfFr = f"{spell_integer(numOfQuadrillions, False)} " if numOfQuadrillions > 1 else ""

    integer_part_spelling = spell_integer(integer_part)
    floating_part_spelling = spell_integer(floating_part)
    hundred_point = HUNDREDS_AS_WORDS[10 ** exponent]
    prefix = HUNDREDS_PREFIX[10 ** exponent]

    return f"{integer_part_spelling} {POINT_AS_WORD} {numOfFr}{hundred_point}{prefix} {floating_part_spelling}"