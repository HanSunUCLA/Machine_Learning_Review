# Machine Learning Review 

Han Sun

## Deal with Missing Data

- Remove missing data rows
- Impute missing values: 1) use a constant value to distinguish missing for missing features, such as 0; 2) use a randomly sampled value based on this feature's distribution; 3) use a mean, median or mode value of this feature; 4) use a value predicted by another model

## Deal with Overfitting

Ways to detect: training/testing split

- Model-wise: 1) use regularization models: reduce variance by applying feature selection models, LASSO, ridge, random forest; 2) use k-fold cross validation; 3) apply ensemble models, Bagging, Boosting, a soft-max layer
- Data-wise: 1) add more data; 2) use data augmentation (deep learning)
- Deep learning: 1) early stopping; 2) drop-out 3) add regularizer for weights

## Identify Outliers

- Extreme value analysis: plot the histogram of individual features and exclude data points that is 3 standard deviation away if the data is Gaussian like. 
- Cluster based: use k-mans to cluster data. Exclude data points that are too far away from its centroid. 
- Use robust models such as LASSO, ridge and decision tree.

## Mostly Used models

### Logistic Regression

$log(\frac{p(y=1)}{1-p(y=1)}) = \beta^{T}X$ implies a sigmoid activation function, which can be seen as $p(y=1) = \frac{1}{1+e^{-\beta^TX}} \in (0, 1)$ . The decisions boundary of it is 0.5.Cost function is cross-entropy: $$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}[ y^{i}log(h_{\beta}(X^{i}) + (1-y^{i}log(1-h_{\beta}(X^{i}))]$$

It is necessary to perform feature selection (remove uncorrelated features, filter out highly correlated features) since the model itself does not enforce it. 

Pros: very efficient for large amount of data solving by gradient descent

Cons: need lots of data to scale well

### Decision Tree

Pro: easy to understand, have value even with small amount of data and is a white box;

Con: unstable, very high variance depending on training data. it is often inaccurate.

### Bagging

Among all those training examples, randomly pick a subset from them with replacement and train decision tree for B times (usually called bootstrap aggregating). The final model is evaluated by averaging prediction of all B trees (regression) or majority vote (classification). 

### Random Forest

Same as Bagging except for each tree, the features are also randomly selected (typically $\sqrt{p}$ features to select) to avoid highly correlated trees in final pool to avoid high variance. Use cross-validation to determine number of trees. 

It can also be used to evaluate feature importance. To measure the importance of the j-th feature after training, the values of the j-th feature are permuted (randomly assigned) among the training data and the out-of-bag error is again computed on this perturbed data set. The importance score for the j-th feature is computed by averaging the difference in out-of-bag error before and after the permutation over all trees. The score is normalized by the standard deviation of these differences.

### Support Vector Machine

For linearly separable data, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The region bounded by these two hyperplanes is called the "margin", and the maximum-margin hyperplane is the hyperplane that lies halfway between them, geometrically as: $max \frac{2}{|w|}$

For non-separable data, use a soft-margin, namely hinge loss: $min \bigg[ \frac{1}{n}\sum_{i=1}^{n}max(0, 1-y_i(w^Tx_i-b))\bigg]+\lambda|w|^2$, parameter  $\lambda$  determines the trade-off between increasing the margin size an sure each data point lies at its correct region. Thus with a small enough $\lambda$ the model is similar to hard-margin SVM.

### K-Means

Unsupervised learning, a variant of generalized EM-algorithm.

1. Initialize random k nodes to be centroid of each cluster;
2. Go through every node and assign them to the k cluster based on some distance measure, e.g., Euclidean distance, Manhattan distance (sum of the absolute differences of their Cartesian coordinates);
3. Recalculate centroid of each cluster and repeat 1, 2 until convergence. 

Elbow to determine k: plot sum of squares error vs. k, visualize an arm like figure and pick the elbow point as optimal k.

Pro: beat hierarchical clustering with tight cluster; it is faster with small k;

Con: global minimum not guaranteed. it may converge to local minimum.

### K Nearest Neighbors

Can be both regression (average value of its k-nearest neighbors) and classification (majority vote of its k neighbors). It is the simplest ML algorithm. 

Pro: simple and fast

Con: it is sensitive to local data structure due to majority vote. 

How to improve: make it inversely weighted by distance to overcome skew data distribution.

**Selection of k**: larger values of *k* reduces effect of the noise on the classification, but make boundaries between classes less distinct. 

1-NN: the bias is low, because you fit your model only to the 1-nearest point. This means your model will be really close to your training data; the variance is high, because optimizing on only 1-nearest point means that the probability that you model the noise in your data is really high. Following your definition above, your model will depend highly on the subset of data points that you choose as training data. If you randomly reshuffle the data points you choose, the model will be dramatically different in each iteration. Basically very high model complexity.

### DBSCAN

An unsupervised classification clustering method. Points are classified as core points, density-reachable points and outliers.

1. A point p is a core point if at least minPts points are within distance ε (ε is the maximum radius of the neighborhood from p) of it (including p). Those points are said to be *directly reachable* from p;
2. A point q is directly reachable from p if point q is within distance ε from point p and p must be a core point;
3. A point q is reachable from p if there is a path *p*1, ..., *pn* with *p*1 = *p* and *pn* = *q*, where each pi+1 is directly reachable from pi (all the points on the path must be core points, with the possible exception of q);
4. All points not reachable from any other point are outliers.

A cluster then satisfies two properties: 1) all points within the cluster are mutually density-connected; 2) if a point is density-reachable from any point of the cluster, it is part of the cluster as well.

### Naive Bayes Classifier

It is a probabilistic classifier based on applying Bayes' theorem and a strong assumption that all features are independent. 

A class's prior, $P(male)$, may be calculated by assuming equiprobable classes, or by calculating an estimate for the class probability from the training set (i.e., (prior for a given class) = (number of samples in the class) / (total number of samples)). The assumptions on distributions of features are called the event model of the Naive Bayes classifier: $P(X_1=x|male) = \frac{1}{\sqrt 2\pi \sigma^2} exp\bigg( \frac{-(x-\mu)^2}{2\sigma^2}\bigg)$

For example, $posterior(male|X) = \frac{P(male)\times P(X_1|male) \times P(X_2|male)}{P(X)}$

### PCA

Rotate the data to project the original feature into a new space where all features are orthogonal and features are ranked by maximum variance. $X_{n\times p}$, $(X^{T}X)_{p\times p}$, therefore the features can be at most p.

## Computer Vision

### Convolutional Neural Network



