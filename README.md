# TensorFlow dataset benchmarks 

Benchmarks of popular Tensorflow datasets with models built from scratch.

## Image classification

| Dataset               |Accuracy: train data[%]| Accuracy: test data[%] | Remarks |
|-----------------------|----------------------|-------------------------|---------|
| _MNIST_               |           99.93      |          98.76          |    -    |
| _FASHION Mnist_       |           95.03      |          91.05          |    -    |
| _horses_or_humans_    |           99.81      |          88.67          |    -    |
| _food101_             |           -          |          -              |    -    |
| _rock_paper_scissors_ |           100        |          87.10          |    -    | 
| _tf_flowers_          |           84.78      |          82.97         |    resnet architecture, mixed precision training    |

## NLP

| Dataset               | Accuracy: train data | Accuracy: test data | Remarks |
|-----------------------|----------------------|---------------------|---------|
| _goemotions_          |           67.17      |          45.38      |    Highly unbalanced dataset. NLP augmentation used. Precision in the test dataset: 61.90%   |
| _imdb_reviews_        |           98.05      |          83.20      |    -    |
