# DictionaryPreTagger
Project for dictionary expansion and word tagging

## Installing

The project needs Keras (possibly with tensorflow backend), requests, python-Levenshtein to run.

The project requires an Apache Solr core running. The [upload.xml](https://drive.google.com/open?id=1aA4n-eIwl-NhdPPN2yz5tykxVMgsuhtl) file should be used to initialize the data.

The project also requires files for the [Bi-LSTM](https://drive.google.com/open?id=12SFzJENppmlfwzpsX2wpMbm7HStDlSQR) and [Convolutional](https://drive.google.com/open?id=1aWD3WOOQKP08Hg11Tc1O5zGt9diyrRhS) neural models in the /Model directory.

The /Tagging/example.py file shows typical usage.
