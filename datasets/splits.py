TEST = [2, 7, 11, 16, 18, 19]

TRAIN = [0, 4, 9, 14]

VAL = [i for i in range(20) if i not in TEST and i not in TRAIN]
