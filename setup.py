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

from __future__ import absolute_import, print_function, division

import os

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'six',
    'scikit-learn',
    'pandas'
]

setup_requires = [
    'numpy',
    'recommonmark'
]

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)


setup(name='trash-pandas',
      version='0.0.1',  
      description='Seamless integration of DataFrames in scikit-learn Pipelines',
      url='http://github.com/dwilson1988/trash-pandas',
      author='Daniel Wilson',
      author_email='harenil@gmail.com',
      license='MIT',
      packages=[package for package in find_packages() if package.startswith('trash')],
      install_requires=install_requires,
      setup_requires=setup_requires,
      zip_safe=False)
