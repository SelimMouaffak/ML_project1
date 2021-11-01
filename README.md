# ML Project 1 : The Higgs Boson

The Higgs Boson is an elementary particle in the standard model of particle physics.
Even though its existence was theorized, it was only observed recently in the CERN
collider in Switzerland. This machine learning project uses data from the CERN and 
aims at training an algorithm that can identify whether the particle is a Higgs
Boson or not. 


# What it does

This code works on a dataset that is based on real data from the CERN collider.
It performs data pre-processing, then trains a specific model on the data. 
The model is then used on new unobserved data to predict whether the particle
is a Higgs Boson or not. 
A lot of different models were used before finding the optimal one for our case.
All the models that we tried can be found in the file implementations.py
This file contains only the algorithms, to try it, place the file in your 
working directory and do the following:

```python
from implementations import *
```

Then all the methods will be available for usage.
Finally, based on the testing set, our implementation
has an accuracy of 82.3%


# How to use 

This code requires a train and test set as input as well as an ouput path to know where to generate results.
You will need to edit the code in order to put the paths of the data relative to your computer.
Moreover, some external function were writter in a seperate file, proj1_helpers.py
You will need this file to be in the same directory as the executable file.
Once it is done, open the terminal in the directory where the executable file is, and type:

```bash
python run.py
```


# Credits

This project was realized by three master students at EPFL, namely : Atallah Amine, Dhraief Mohamed Ali
and Mouaffak Selim.
