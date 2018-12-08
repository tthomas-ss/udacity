This project was implemented on an AWS EC2 instance of type Deep Learning AMI (Ubuntu) Version 19.0.
All Keras RNN models must be run on GPU, as these use the CuDNN version of GRU and LSTM layers.

A requirements.txt file is included which contains everything included in the tensorflow_p36 environment on the AWS instance.

Data and embeddings can be found here: https://www.kaggle.com/c/quora-insincere-questions-classification/data
All submitted code is in ./submit/:

* 00.Explore-dataset.ipynb
* 01.Benchmark Logistic Regression.ipynb
* 02.FFNN.ipynb
* 03.SimpleRNN.ipynb
* 04.CNN.ipynb
* 05.GRU-Onedirectional.ipynb
* 06.GRU.ipynb
* 08.LSTM Final.ipynb
* capstone_utils.py

The following main libraries were used:

|    Library            |    Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    scikit-learn       |    Open source library for machine learning, built on Numpy, SciPy and Matplotlib   (see below).  Contains tools for   classification, regression, clustering, dimensionality reduction, model   selection and pre-processing.  This   project will use scikit-learn for classification (logistic regression and   random forest), model selection (grid search and metrics), feature extraction   and pre-processing (e.g. splitting training data in training and validation).    |
|    Keras              |    Open source high-level (i.e. sacrificing some control for simplicity) neural   networks API, written in Python and capable of running on top of several   underlying neural network frameworks.  Its   high-level API makes it very suitable to prototype different neural network   architectures.  This project will use   Keras with a Tensorflow   backend for developing neural networks and NLP tasks.                                                                     |
|    NLTK               |    Natural Language Toolkit.  A suite of libraries   for classification, tokenization, stemming, tagging, parsing and semantic   reasoning.  This project will use NLTK   for NLP tasks such as tokenization, word-to-vector conversion, stop word-removal   and lemmatization.                                                                                                                                                                                                     |
|    Spacy              |    Library for NLP in Python, with much overlap with NLTK.  This project will use NLTK for NLP tasks   such as tokenization, word-to-vector conversion, stop word-removal and   lemmatization.                                                                                                                                                                                                                                                                                      |
|    Pandas             |    Pandas is a Python data analysis library, providing fast, flexible data structures for working with data.  Pandas will be used   to hold data structures as well as reading/writing/parsing files and cleaning   data.                                                                                                                                                                                                                                                         |
|    Numpy              |    Python library for scientific programming, adding support for large   multi-dimensional arrays and matrices and many high-level mathematical   functions.  Numpy is required for several   of the other libraries used and will be used by itself for vector/matrix   operations.                                                                                                                                                                                                |
|    Matplotlib         |    Python plotting library for programmatic visualization.  This will be used for all visualizations.                                                                                                                                                                                                                                                                                                                                                                                 |