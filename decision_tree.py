import numpy as np

# define node class
class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value != None

# define decision tree class
class DecisionTree():
    # initialisation method (can only set max depth and min samples/node)
    def __init__(self, max_depth=30, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = None
        self.root = None
    
    # fit method is used for training
    def fit(self, X, y):
        # set attributes that cant be passed
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X,y)
    
    # helper fx to grow tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # print(f"depth: {depth}, n: {n_samples}, f: {n_features}")
        unique_labels, unique_labels_counts = np.unique(y, return_counts=True)
        # print(unique_labels)
        
        # check stopping condition
        if (len(unique_labels)==1 or depth>self.max_depth or n_samples<=self.min_samples):
            leaf_value = unique_labels[np.argsort(unique_labels_counts)[-1]]
            return Node(value=leaf_value)
        
        ftr_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # find best split
        best_ftr_idx, best_ftr_thr, no_gain = self._best_split(X, y, n_samples, ftr_idxs)
        
        if no_gain:
            leaf_value = unique_labels[np.argsort(unique_labels_counts)[-1]]
            return Node(value=leaf_value)
        
        # grow children
        left_child_idxs, right_child_idxs = np.where(X[:,best_ftr_idx]<=best_ftr_thr)[0], np.where(X[:,best_ftr_idx]>best_ftr_thr)[0]
        left = self._grow_tree(X[left_child_idxs], y[left_child_idxs], depth+1)
        right = self._grow_tree(X[right_child_idxs], y[right_child_idxs], depth+1)
        return Node(best_ftr_idx, best_ftr_thr, left, right)
    
    # helper fx to find the best split at a node
    def _best_split(self, X, y, n_samples, ftr_idxs):
        no_gain = False
        best_gain = -1
        best_ftr_idx, best_ftr_thr = None, None
        for ftr_idx in ftr_idxs:
            # i missed this point, thr in thrs should be unique
            thrs = np.unique(X[:,ftr_idx])
            for thr in thrs:
                gain = self._info_gain(X,y,thr,ftr_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_ftr_idx = ftr_idx
                    best_ftr_thr = thr
        # print(best_gain)
        if best_gain == 0.0:
            no_gain = True
        return best_ftr_idx, best_ftr_thr, no_gain
    
    # helper fx to calculate information gain
    def _info_gain(self, X, y, thr, ftr_idx):
        
        # parent entropy
        parent_entropy = self._entropy(y)
        
        # instantiate children
        left_child_idxs, right_child_idxs = np.where(X[:,ftr_idx]<=thr)[0], np.where(X[:,ftr_idx]>thr)[0]
        left_child_weight, right_child_weight = len(left_child_idxs)/len(y), len(right_child_idxs)/len(y)
        left_child_entropy, right_child_entropy = self._entropy(y[left_child_idxs]), self._entropy(y[right_child_idxs])
        
        # weighted children entropy
        weighted_children_entropy = left_child_weight*left_child_entropy + right_child_weight*right_child_entropy
        
        # calculate info gain
        return parent_entropy - weighted_children_entropy
    
    # helper fx to calulate entropy
    def _entropy(self, y):
        y = y.astype(int)
        e = 0
        ps = np.bincount(y)/len(y)
        for p in ps:
            if p != 0:
                e += p * np.log(p)
        return -e
    
    # predict method is used for testing
    def predict(self, X):
        
        y_pred = np.array([])
        for x_i in X:
            y_pred = np.append(y_pred, np.array([self._traverse_tree(x_i, self.root)]))
        return y_pred
    
    # helper fx to traverse the tree
    def _traverse_tree(self, x_i, node):
        if node.is_leaf():
            return node.value
        if x_i[node.feature] <= node.threshold:
            return self._traverse_tree(x_i, node.left)
        return self._traverse_tree(x_i, node.right)
    
# End of Part 2