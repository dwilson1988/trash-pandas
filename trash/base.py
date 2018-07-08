###################################################################################################
# MIT License
#
# Copyright (c) 2018 Daniel WIlson
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

from __future__ import print_function, absolute_import, division
from six import with_metaclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

class PDTransformerMetaClass(type):
    def __call__(cls,*args,**kwargs):
        print(args,kwargs)
        kwargs.pop('yo')
        instance = cls.__new__(cls,*args,**kwargs)
        instance.__init__(*args,**kwargs)
        return instance

class BaseTransformer(with_metaclass(
    PDTransformerMetaClass,BaseEstimator,TransformerMixin)):
    _overrides = ('fit','transform','fit_transform')

    def super_getattr(self,attr):
        return super(BaseTransformer,self).__getattribute__(attr)

    def _fit_transform(self,X,y=None,**fit_params):
        return self.super_getattr('fit_transform')(X,y=y,**fit_params)

    def _transform(self,X,**transform_params):
        Xt = self.super_getattr('transform')(X,**transform_params)
        return pd.DataFrame(Xt,columns=self.column_names)

    def transform(self,X,**transform_params):
        return X

    def fit(self,X,y=None,**fit_params):
        """
        To be overriden. Implement the logic needed for the fit method of your transformor, minus 
        the logic that this library will take care of.
        """
        return self

    def _fit(self,X,y=None,**fit_params):
        self.column_names = X.columns

        return self.super_getattr('fit')(X,y=y,**fit_params)

    def __getattribute__(self,attr):
        """
        We enforce certain behavior on the "special" methods. Anyone who overrides this class 
        can override the 'fit', 'transform', and 'fit_transform' methods. This class will redirect
        the call of this to handle the additional logic required for DataFrames that will not need
        to be implemented by the child class.
        """
        if attr in super(BaseTransformer,self).__getattribute__('_overrides'):
            return super(BaseTransformer,self).__getattribute__('_'+attr)
        return super(BaseTransformer,self).__getattribute__(attr)
