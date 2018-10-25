###################################################################################################
# MIT License
#
# Copyright (c) 2018 Daniel Wilson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################################

import os
import importlib
import warnings

import sklearn.preprocessing as sp

from trash.base import BaseTransformer

__all__ = ['BaseTransformer','patched','patch']

# Define the patched methods. This can be imported directly or patched into scikit learn
class StandardScaler(sp.StandardScaler, BaseTransformer): pass
class Binarizer(sp.Binarizer, BaseTransformer): pass
class FunctionTransformer(sp.FunctionTransformer, BaseTransformer): pass
class PowerTransformer(sp.PowerTransformer, BaseTransformer): pass
class MaxAbsScaler(sp.MaxAbsScaler, BaseTransformer): pass
class MinMaxScaler(sp.MinMaxScaler, BaseTransformer): pass
class RobustScaler(sp.RobustScaler, BaseTransformer): pass
class Normalizer(sp.Normalizer, BaseTransformer): pass
class QuantileTransformer(sp.QuantileTransformer, BaseTransformer): pass

classes_to_patch = {
    'sklearn.preprocessing': {
        'StandardScaler':StandardScaler,
        'Binarizer':Binarizer,
        'FunctionTransformer':FunctionTransformer,
        'PowerTransformer':PowerTransformer,
        'MaxAbsScaler':MaxAbsScaler,
        'MinMaxScaler':MinMaxScaler,
        'Normalizer':Normalizer,
        'QuantileTransformer':QuantileTransformer,
        'RobustScaler':RobustScaler
    }
}

patched = {}

def patch(class_name=None):
    """
    Applies a patch to all scikit learn Transformers to work on pandas DataFrames
    """
    warnings.warn('Calling `trash.patch` modified scikit-learn (runtime),'
        ' be sure you understand the consequences of this action')
    # Loop through 'classes_to_patch'
    for module,classes in classes_to_patch.items():
        patched[module] = []
        # Import the module that contains classes to patch
        mod = importlib.import_module(module)
        # Patch each class with a dynamically created metaclass that inherits from BaseTransformer.
        for cln,cls in classes.items():
            if class_name is not None and cln != class_name:
                continue
            patched[module].append(getattr(mod,cln))
            # This line creates a new metaclass (type) and replaces the original class in the module.
            setattr(mod,cln,cls)

def unpatch(class_name=None):
    """
    Reverses the patch by replacing classes with the originals
    """
    for module,classes in patched.items():
        
        # Import the module that contains classes to patch
        mod = importlib.import_module(module)
        # Patch each class with a dynamically created metaclass that inherits from BaseTransformer.
        for cls in classes:
            if class_name is not None and cls != class_name:
                continue
            # This line creates a new metaclass (type) and replaces the original class in the module.
            setattr(mod,cls.__name__,cls) 


