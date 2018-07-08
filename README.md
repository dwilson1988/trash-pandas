<img src="images/logo.png" width="100" height="100">

# trash-pandas

## Introduction: What is `trash-pandas`?
A helper library for integrating `pandas` and `scikit-learn`. It contains all the trash that everyone writes, but for some reason no one has really consolidated all that well into a library. 

Both `scikit-learn` and `pandas` are terrific libraries. `scikit-learn`'s `Pipeline` and `
FeatureUnion` classes make it easy to prepare data for machine learning by the use of `Transformers`. 
You can perform end to end data prep, training, and validation - unless you also want to do all of 
that with a `pandas` `DataFrame`. In addition, pandas has convenient utilities for handling missing
data, but all of that goes out the door when trying to use scikit-learn `Transformers` such as the
`StandardScaler`. As both libraries are ubiquitous in the data science community, it seems odd that 
there are very few options to integrate the two. Most users end up writing or reusing some design 
patterns that implement simple transformers on pandas DataFrames such as selection by index/column, 
dtype, etc.

`trash-pandas` is our attempt to unify the two with a lightweight and extensible wrapper on the 
scikit-learn's `Transformer` concept.  `trash-pandas` provides some simple base classes that wrap 
scikit-learn's functionality so that users can easily write their own `DataFrame` friendly 
`Transformers` as well as some `Transformers` that users commonly use.

"Trash panda" is of course a nickname given to racoons by some people. 

## What **isn't** `trash-pandas`?

This is not an attempt to reproduce the work of `sklearn-pandas` or `pandas-ml`, both of which take a very different approach. 
`trash-pandas` attempts to build a very light and thin layer to weave together `pandas` and `scikit-learn` in the most flexible way 
possible so that users can do things mostly the same as always, but with less boilerplate and more flexibility. `pandas-ml` offers 
a full integration of the two libraries and `sklearn-pandas` offers a way to build `DataFrame` friendly pipelines.
