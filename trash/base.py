import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

class BaseTransformer(BaseEstimator, TransformerMixin):
    _overrides = ('fit','transform','fit_transform')

    def __init__(self):
	    pass

    def _fit_transform(self,X,y=None,**fit_params):
        return super(BaseTransformer,self).__getattribute__('fit_transform')(X,y=y,**fit_params)

    def _transform(self,X,**transform_params):
        return super(BaseTransformer,self).__getattribute__('transform')(X,**transform_params)

    def transform(self,X,**transform_params):
        return X

    def fit(self,X,y=None,**fit_params):
        return self

    def _fit(self,X,y=None,**fit_params):
        return super(BaseTransformer,self).__getattribute__('fit')(X,y=y,**fit_params)

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
