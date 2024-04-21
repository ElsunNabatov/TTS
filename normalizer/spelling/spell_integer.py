# Constants
from normalizer.utils.helpers import ZERO, TEN, HUNDRED, THOUSAND, MILLION, BILLION, TRILLION, QUADRILLION

# Translations
from normalizer.utils.translations import DIGITS_AS_WORDS, TENTHS_AS_WORDS, NEGATIVE_AS_WORD, HUNDRED_AS_WORD, THOUSAND_AS_WORD, MILLION_AS_WORD, BILLION_AS_WORD, TRILLION_AS_WORD, QUADRILLION_AS_WORD


# Cache
cache = {}

def spell_integer(input, spellZeroAtTheEnd=True):
    global cache

    if input in cache:
        return cache[input]

    spelling = ''
    sign = NEGATIVE_AS_WORD + " " if input < 0 else ''
    number = abs(input)

    if input == ZERO and spellZeroAtTheEnd:
        zero = DIGITS_AS_WORDS[0]
        cache[input] = zero
        return zero

    if ZERO < number < TEN:
        spelling_of_digit = DIGITS_AS_WORDS[number]
        cache[number] = spelling_of_digit
        spelling += spelling_of_digit
    elif TEN <= number < HUNDRED:
        number_of_tens = number // TEN
        digit_point = " " + DIGITS_AS_WORDS[number % TEN] if number % TEN > 0 else ''
        final_spelling = TENTHS_AS_WORDS[number_of_tens * TEN] + digit_point
        cache[number] = final_spelling
        spelling += final_spelling
    elif HUNDRED <= number < THOUSAND:
        number_of_hundreds = number // HUNDRED
        remainder = number % HUNDRED
        num_of_hundreds_spelling = spell_integer(number_of_hundreds) if number_of_hundreds > 1 else ''
        hundreds_spelling = num_of_hundreds_spelling + " " + HUNDRED_AS_WORD if num_of_hundreds_spelling else HUNDRED_AS_WORD
        final_spelling = hundreds_spelling + " " + spell_integer(remainder, False) if remainder > 0 else hundreds_spelling
        cache[number] = final_spelling
        spelling += final_spelling
        
    # for THOUSAND, MILLION, BILLION, TRILLION, and QUADRILLION cases
    elif THOUSAND <= number < MILLION:
        number_of_thousands = number // THOUSAND
        remainder = number % THOUSAND
        num_of_thousands_spelling = spell_integer(number_of_thousands) if number_of_thousands > 1 else ''
        thousands_spelling = num_of_thousands_spelling + " " + THOUSAND_AS_WORD if num_of_thousands_spelling else THOUSAND_AS_WORD
        final_spelling = thousands_spelling + " " + spell_integer(remainder, False) if remainder > 0 else thousands_spelling
        cache[number] = final_spelling
        spelling += final_spelling

    elif MILLION <= number < BILLION:
        number_of_millions = number // MILLION
        remainder = number % MILLION
        num_of_millions_spelling = spell_integer(number_of_millions) if number_of_millions > 1 else ''
        millions_spelling = num_of_millions_spelling + " " + MILLION_AS_WORD if num_of_millions_spelling else MILLION_AS_WORD
        final_spelling = millions_spelling + " " + spell_integer(remainder, False) if remainder > 0 else millions_spelling
        cache[number] = final_spelling
        spelling += final_spelling

    elif BILLION <= number < TRILLION:
        number_of_billions = number // BILLION
        remainder = number % BILLION
        num_of_billions_spelling = spell_integer(number_of_billions) if number_of_billions > 1 else ''
        billions_spelling = num_of_billions_spelling + " " + BILLION_AS_WORD if num_of_billions_spelling else BILLION_AS_WORD
        final_spelling = billions_spelling + " " + spell_integer(remainder, False) if remainder > 0 else billions_spelling
        cache[number] = final_spelling
        spelling += final_spelling

    return sign + spelling