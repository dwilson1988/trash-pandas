from functools import wraps

import numpy as np

def arraycheck(f):
    """
    Decorator defaults to default behavior if this is applied to an
    ndarray rather than a DataFrame
    """
    
    method_name = f.__name__
    
    @wraps(f)
    def wrapped(*args,**kwargs):
        self = args[0]
        X = args[1]

        if isinstance(X,np.ndarray):
            return self.super_getattr(method_name[1:])(*args[1:],**kwargs)
        return f(*args,**kwargs)
    
    return wrapped
