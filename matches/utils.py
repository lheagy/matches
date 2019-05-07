
def set_kwargs(obj, ignore=None,  **kwargs):
    """
    Sets key word arguments (kwargs) that are present in the object,
    throw an error if they don't exist.
    """
    if ignore is None:
        ignore = []
    for attr in kwargs:
        if attr in ignore:
            continue
        if hasattr(obj, attr):
            setattr(obj, attr, kwargs[attr])
        else:
            raise Exception(f"{attr} attr is not recognized")
