# Isolation_forest
## Goal:
This project is an implementation of Isolation Forest algorithm by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.

## Algorithm:
Anomalies are few and different. Isolation forest algorithm focus on those anomaly points. 
See the scripts for the details. [Scripts](https://github.com/ziyufan/Isolation_forest/blob/master/scripts/iforest.py)

For your convenience, here are the algorithms extracted from the Liu et al paper:
<p float="left">
  <img src="https://github.com/ziyufan/Isolation_forest/blob/master/image/iForest.png" width="500" />
  <img src="https://github.com/ziyufan/Isolation_forest/blob/master/image/iTree.png" width="500" /> 
  <img src="https://github.com/ziyufan/Isolation_forest/blob/master/image/PathLength.png" width="500" />
</p>


## Example:
Fit the Isolation tree algorithm to cancer dataset. Here is the anomaly score threshold for the model. Checkout the notebook for more details about the example. [Example](https://github.com/ziyufan/Isolation_forest/blob/master/exp/Isolation_Tree_Anomaly_Detection.ipynb)

![alt text](https://github.com/ziyufan/Isolation_forest/blob/master/image/result.png "Part 1. iTree")

## Reference:
Isolation-based Anomaly detection, Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou [Original paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.673.5779&rep=rep1&type=pdf)
