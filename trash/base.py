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

from __future__ import print_function, absolute_import, division
from six import with_metaclass

import dill
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import Imputer

from trash.utils import arraycheck

class PDTransformerMetaClass(type):
    """
    This class inherits from a metaclass that manages parameters that shouldn't 
    get passed to the constructor, but rather get managed by the wrapper 
    BaseTransformer class wrapping the transformer. In addition to the original 
    object's constructor, the object may optionally be provided the following
    arguments.

    Args:
        columns (list of str or str): Which columns to apply this transformer on
            or 'all' to specify use all columns. Default 'all'
        na (str): How to handle NA values. 'ignore' will apply the 
            transformer and leave na values in place. If 'drop', will drop rows 
            containing any NA value. If 'impute', will replace values with specified
            imputation scheme through the 'impute' argument. If 'raise', a exception
            will be raised. 
        impute_method (str, function, or dict of str or function): How to replace missing 
            values, if na=='impute'. If it's a string, can be one of 'mean', 
            'median', 'mode', or 'zeros'. If it's a function, 
            function must take a ``pd.Series`` (a column) and optional keyword 
            arguments (passed in by 'impute_args' argument). If a dictionary, 
            specify the imputation method as a str or function for each column.
        impute_args (tuple or dict of tuples): arguments to user specified impute 
            function(s). If a dict, specifies individual arguments for each column.
        return_values (bool): If True, returns only the values. This is useful if 
            trash.patch() has been called, which replaces scikit-learn transformers 
            with ``pd.DataFrame`` friendly ones. Set to True if you want numpy arrays.

    Examples:
        >>> class DFStandardScaler(StandardScaler, BaseTransformer): pass
        ...
        >>> scaler_drop = DFStandardScaler(columns=['A','B'],na='ignore')
        >>> scaler_impute = DFStandardScaler(columns='all',na='impute',impute='mean')
    """

    def __call__(cls,*args,**kwargs):
        """
        This metaclass takes care of object construction. It's important because
        there are arguments that this metaclass can receive that are not valid
        when wrapping, say, a scikit-learn transformer that takes no parameters.
        This allows the functionality of this library to extend transformers
        without overriding the constructor.
        """
        # Extract the kwargs that trash-pandas uses.
        trash_kwargs = {
            'columns': kwargs.pop('columns','all'),
            'na': kwargs.pop('na','raise'),
            'impute_method': kwargs.pop('impute_method','raise'),
            'impute_args': kwargs.pop('impute_args',()),
            'return_values': kwargs.pop('return_values',False)
        }

        # Create instance using remaining kwargs
        instance = cls.__new__(cls,*args,**kwargs)
        instance.__init__(*args,**kwargs)

        # Intialize to unmasked na values
        setattr(instance,'where_mask',None)
        # Output columns will be the same as input columns unless otherwise specified.
        setattr(instance,'columns_out',None)

        # Apply attributes to the instance. These will be available to BaseTransformer
        for k,v in trash_kwargs.items():
            setattr(instance,k,v)

        if instance.na == 'impute' and instance.impute_method is None:
            raise ValueError("'na' set to 'impute', but 'impute_method' is None!")
        if (not isinstance(instance.columns,list)) and \
           (not isinstance(instance.columns,str)):
            raise ValueError("'columns' must be a list of column names or 'all'!")
       
        # Return the instance, constructor has already been called.
        return instance
    

class BaseTransformer(with_metaclass(
    PDTransformerMetaClass,BaseEstimator,TransformerMixin)):
    _overrides = {'fit','transform','fit_transform'}

    def super_getattr(self,attr):
        return super(BaseTransformer,self).__getattribute__(attr)

    @arraycheck
    def _fit_transform(self,X,y=None,**fit_params):
        if self.columns == 'all':
            self.columns = X.columns.tolist()
        if self.columns_out is None:
            self.columns_out = self.columns

        return self.super_getattr('fit_transform')(X,y=y,**fit_params)

    @arraycheck
    def _transform(self,X,**transform_params):
        Xt,_ = self._select(X)

        Xt,yt = self._handle_na(Xt,None)

        Xt = self.super_getattr('transform')(Xt,**transform_params)

        return self._as_dataframe(Xt)

    def transform(self,X,**transform_params):
        return X

    def fit(self,X,y=None,**fit_params):
        """
        To be overriden. Implement the logic needed for the fit method of your transformor, minus 
        the logic that this library will take care of.
        """
        return self

    @arraycheck
    def _fit(self,X,y=None,**fit_params):
        if self.columns == 'all':
            self.columns = X.columns.tolist()
        if self.columns_out is None:
            self.columns_out = self.columns
            
        Xt,yt = self._select(X,y=y)
        Xt,yt = self._handle_na(Xt,yt)
        return self.super_getattr('fit')(Xt,y=yt,**fit_params)

    def _handle_na(self,X,y):
        return getattr(self,'_na_'+self.na)(X,y)

    def _na_ignore(self,X,y):
        return X,y

    def _na_drop(self,X,y):
        idx = ~X.isna(axis=0)
        if y is None:
            return X.dropna()
        return X.dropna(),y.loc[idx]

    def _na_raise(self,X,y):
        if X.isna().sum():
            raise ValueError("In transformer `{}`. Was passed DataFrame with NA values!".format(
                self.__class__.__name__))
        return X,y

    def _na_impute(self,X,y):
        if self.impute_method in ['mean','median','most_frequent']:
            value = getattr(X,self.impute_method)(axis=0)
            return X.fillna(value),y
        
        elif self.impute_method == zeros:
            return X.fillna(0),y
        
        elif callable(self.impute_method):
            Xt = X.copy()
            for column in self.columns:
                Xt.loc[:,column] = self.impute_method(X.xs(column,axis=1))
            return Xt,y
        elif isinstance(self.impute_method,dict):
            Xt = X.copy()
            for column,func in self.impute_method.items():
                Xt.loc[:,column] = func(X.xs(column,axis=1))
        else:
            return X,y

    def _select(self,X,y=None):
        """
        Selects specific subset of the data, for example rows not containing nan
        """
        return X.loc[:,self.columns], y

    def _as_dataframe(self,X):
        return pd.DataFrame(X,columns=self.columns_out)

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


class TypeSelector(BaseTransformer):
    """
    The simple pattern that selects data based on their type to work with mixed type DataFrames
    """
     
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
    
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


