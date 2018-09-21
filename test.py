from __future__ import absolute_import

import trash
trash.patch()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.DataFrame({
    'a':[1.0,2.0,3.0,4.0,5.0],
    'b':[0,1,1,1,16]
})

ss = StandardScaler()

print(ss.columns,ss.na,ss.impute_method,ss.impute_args,ss.return_values)

X = np.random.randn(25,5)
#print(ss.fit_transform(X))

ss.fit(df)
print(ss.fit_transform(df))
print(ss.fit_transform(X))
