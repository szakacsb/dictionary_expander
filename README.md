# DictionaryPreTagger
Project for dictionary expansion and word tagging

## Installing

The project needs Keras (possibly with tensorflow backend), requests, python-Levenshtein to run.

The project requires an Apache Solr core running. The [upload.xml](https://drive.google.com/open?id=1aA4n-eIwl-NhdPPN2yz5tykxVMgsuhtl) file should be used to initialize the data.

The project also requires files for the [Hybrid](https://drive.google.com/file/d/1GTmrqkYGqa3p5bHjjJfUl3wABbDF2E_l/view?usp=sharing) and [Convolutional](https://drive.google.com/file/d/19chzUZ-hPXu0H3rD5CBI3I6Pv7T6SBmB/view?usp=sharing) neural models in the /Model directory.

The /Tagging/example.py file shows typical usage.
