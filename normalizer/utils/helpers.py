import re

def isInteger(number):
    """
    Checks if number is integer or not
    :param number: arbitrary number
    :return: boolean
    """
    return number == int(number)

# Maximum value that we can spell for now
MAX_SAFE_VALUE = 9007199254740991

CURRENCY_SYMBOLS = {
    "₼": "manat",
    "AZN": "manat",
    "$": "dollar",
    "USD": "dollar",
    "€": "avro",
    "EUR": "avro",
    "£": "funt",
    "GBP": "funt",
    "₽": "rubl",
    "RUB": "rubl",
    "₺": "lira",
    "TRY": "lira"
}

MONTHS_AS_WORDS = {
    1: "yanvar",
    2: "fevral",
    3: "mart",
    4: "aprel",
    5: "may",
    6: "iyun",
    7: "iyul",
    8: "avqust",
    9: "sentyabr",
    10: "oktyabr",
    11: "noyabr",
    12: "dekabr"
}

ZERO = 0
TEN = 10
HUNDRED = 100
THOUSAND = 1000
MILLION = 1000000
BILLION = 1000000000
TRILLION = 1000000000000
QUADRILLION = 1000000000000000

def find_numbers_and_currencies_in_text(text):
    matches = re.findall(r'([^\d.\s]+)?\s*(\d+\.\d+|\d+)\s*([a-zA-Z$€£¥₹₽₺₴₣₸₪₩₮₵]+)?', text)

    items = []

    for match in matches:
        before, amount, after = match

        before = before.strip() if before else None
        after = after.strip() if after else None

        # Use the currency symbol from the dictionary if it matches, otherwise use None
        symbol = before if before in CURRENCY_SYMBOLS.keys() else after if after in CURRENCY_SYMBOLS.keys() else None

        if amount:
            if '.' in amount:
                amount = float(amount)
            else:
                amount = int(amount)
            items.append((symbol, amount))

    return items

def find_dates_in_text(text):
    matches = re.findall(r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})', text)

    items = []

    for match in matches:
        day, month, year = match
        items.append((day, month, year))

    return items