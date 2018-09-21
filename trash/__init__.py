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

from .base import BaseTransformer


__all__ = ['BaseTransformer','patched','patch']


classes_to_patch = {
    'sklearn.preprocessing': [
        'StandardScaler',
        'Binarizer',
        'FunctionTransformer',
        'MaxAbsScaler',
        'MinMaxScaler',
        'Normalizer',
        'OneHotEncoder',
        'PolynomialFeatures',
        'QuantileTransformer',
        'RobustScaler'
    ]
}

patched = []

def patch():
    """
    Applies a patch to all scikit learn Transformers to work on pandas DataFrames
    """

    # Loop through 'classes_to_patch'
    for module,classes in classes_to_patch.items():
        # Import the module that contains classes to patch
        mod = importlib.import_module(module)
        # Patch each class with a dynamically created metaclass that inherits from BaseTransformer.
        for cls in classes:
            # This line creates a new metaclass (type) and replaces the original class in the module.
            setattr(mod,cls,
                type(cls,(getattr(mod,cls),BaseTransformer),{})
            ) 
    
