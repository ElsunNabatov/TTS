from normalizer.utils.helpers import MONTHS_AS_WORDS
from normalizer.spelling.spell_integer import spell_integer

def spell_date(day, month, year):

    # raise assertion error if day and month numbers are not in range
    assert 1 <= int(day) <= 31, "Gun 1 ve 31 arasinda olmalidir."
    assert 1 <= int(month) <= 12, "Ay 1 ve 12 arasinda olmalidir."

    # Convert day, month, and year into their spelled equivalents
    day_spelled = spell_integer(int(day))
    month_spelled = MONTHS_AS_WORDS[int(month)]
    year_spelled = spell_integer(int(year))

    # Construct the spelled date format
    spelled_date = f"{day_spelled} {month_spelled} {year_spelled}"

    return spelled_date