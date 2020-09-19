from .in1d import in1d


def notin1d(array1, array2):
    return in1d(array1, array2, invert=True)
