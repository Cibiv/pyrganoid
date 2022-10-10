# Pyrganoid

pyrganoid is a python library to simulate multitype branching processing, with
the goal of simulating the growth of organoids. 

# Installation

First install the python packages cython, cythongsl, numpy and pyarrow:

    pip install cython cythongsl numpy pyarrow wheel

Then build the wheel via

    python setup.py bdist_wheel
    
Then install the package via

    pip install dist/pyrganoid-6.0.1-cp310-cp310-linux_x86_64.whl


