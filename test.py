from __future__ import absolute_import

from trash.base import BaseTransformer

import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.DataFrame({
    'a':[1.0,2.0,3.0,4.0,5.0],
    'b':[0,1,1,1,16]
})


class DFStandardScaler(StandardScaler,BaseTransformer):
    pass

print(dir(BaseTransformer))
print(dir(StandardScaler))
ss = DFStandardScaler(yo=True)
print(dir(ss))
print(ss.fit_transform(df))
