"""
The character vocabulary the CNN predicts per grid cell.
"""

# index 0 is blank; the rest are the full befunge instruction set
VOCAB = " 0123456789+*-/%!`><^v?_|\":\\$.,#gp&~@"

CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
