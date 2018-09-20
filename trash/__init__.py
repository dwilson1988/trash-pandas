
__all__ = ['BaseTransformer','patched','patch']


classes_to_patch = []

patched = []

def patch():
    """
    Applies a patch to all scikit learn Transformers to work on pandas DataFrames
    """

    
