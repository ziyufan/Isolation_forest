
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.height_limit = np.ceil(np.log2(sample_size))
        
    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        for i in range(self.n_trees):
            X_sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False), :]
            itree = IsolationTree(self.height_limit)
            itree.fit(X_sample, improved)
            self.trees.append(itree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        path_length = []
        for x_i in X:
            length = []
            for tree in self.trees:
                length.append(tree.path_length_single(x_i))
            path_length.append(np.mean(length))
        return np.array(path_length).reshape(len(X), 1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        expected_path_length = self.path_length(X)
        return 2**(-(expected_path_length/c(self.sample_size)))
    
    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores>=threshold).astype(int)
    
    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.n_nodes = 0
        self.root = None

    def fit(self, X:np.ndarray, e=0, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if e>=self.height_limit or len(X)<=1:
            self.n_nodes = self.n_nodes + 1
            return Tree(X,None,None,None,None,'ex')
        else:
            Q = np.arange(X.shape[1], dtype='int')
            q = np.random.choice(Q)
            q_min = X[:,q].min()
            q_max = X[:,q].max()
            if improved:
                p_list = np.random.uniform(q_min,q_max,5)
                best_p = q_max
                x_len = len(X)
                for p in p_list:
                    X_left = X[np.where(X[:,q] < p)]
                    X_right = X[np.where(X[:,q] >= p)]
                    if min(len(X_left), len(X_right))<=5:
                        best_p = p
                        break
                    if min(len(X_left), len(X_right))<x_len:
                        best_p = p
            else:
                best_p = np.random.uniform(q_min,q_max)
            X_left = X[np.where(X[:,q] < best_p)]
            X_right = X[np.where(X[:,q] >= best_p)]
            self.n_nodes = self.n_nodes + 1
            self.root = Tree(None,q, best_p, self.fit(X_left,e+1), self.fit(X_right,e+1), 'in')
        return self.root
    
    def path_length_single(self, x, e=0):
        p = self.root
        while p.type == 'in':
            if x[p.splitAtt]< p.splitValue:
                p = p.left 
            else:
                p = p.right
            e = e+1
        return e+c(len(p.value))            

class Tree:
    def __init__(self,Value,splitAtt,splitValue,left, right, node_type):
        self.value = Value
        self.splitAtt = splitAtt
        self.splitValue = splitValue
        self.left = left
        self.right = right
        self.type = node_type

def c(x):
    if x>2:
        return 2*(np.log(x-1) + 0.5772156649)-2*(x-1)/x
    elif x==2:
        return 1
    else:
        return 0

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    for threshold in np.arange(1,0,-0.01):
        y_hat = (scores>=threshold).astype(int)
        confusion = confusion_matrix(y, y_hat)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return threshold, FPR

