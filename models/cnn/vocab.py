"""The character vocabulary the CNN predicts per grid cell."""

# index 0 is blank; the rest are the befunge ops the corpus needs
VOCAB = " 0123456789+-*/%><^v_|:\\$.gp@"

CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
