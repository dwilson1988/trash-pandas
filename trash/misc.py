import numpy as np


def arraycheck(f):
    """
    Decorator defaults to default behavior if this is applied to an
    ndarray rather than a DataFrame
    """
    method_name = f.__name__
    def wrapped(*args,**kwargs):
        self = args[0]
        X = args[1]

        if isinstance(X,np.ndarray):
            return self.super_getattr(method_name[1:])(*args[1:],**kwargs)
        return f(*args,**kwargs)
    wrapped.__name__ = method_name
    wrapped.__doc__ = f.__doc__
    return wrapped
