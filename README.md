This section presents three comparative deep learning methods used in the paper.

The code files for each comparative method have similar naming and function. Taking EDHRN's one-dimensional under-sampled data (two-dimensional data) as an example:

demo.py generates simulation data for training; earlystop.py, model.py (model code), and train.py are all used for training; run2d.py is used for testing with real data.

Therefore, code with the .3D extension is used for two-dimensional under-sampled data (three-dimensional data).

(EDHRN is missing a train3D.py training file. You can directly modify the import section of train.py to import the 3D training data. The other two comparative methods have corresponding train3D.py training code.)
