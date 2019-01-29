# Machine Learning Review 

Han Sun, Ph.D. in Earthquake and Structural Engineering, M.S. in Statistics, University of California, Los Angeles



## Learning Theory

### Bias vs. Variance

High bias, underfitting; high variance, overfitting, high model complexity

### Empirical Risk Minimization

Let the hypothesis be: $h: X\to Y$, the risk associated with $h$ is $R(h) = E[L(h(x), y)] = \int L(h(x), y)dP(x, y)$. The goal is to find a hypothesis $h^{*}$ which minimizes it. The empirical risk minimization is defined over a set of training data:

$R_{emp}(h) = \frac{1}{n}\sum_{i=1}^{n}L(h(x_i), y_i)$

Instead of finding $h^{*}​$, find: $\hat{h}​$

### Training Error and Generalization Error

Let $f(x)$ be the prediction model which maps from $x$ to $y$, $L(\cdot, \cdot)$ be a loss measure, the expected error of a particular model $f_n(x)$ is defined over all possible values of $x$ and $y$:  

$I[f_n] = \int_{X \times Y} L(f_n(x), y)p(x, y)dxdy$ 

Without knowing the joint probability of $p(x, y)$, we can only compute the empirical error over training dataset:

$$I_S[f_n] = \frac{1}{n}\sum_{i=1}^{n}L(f_n(x_i), y_i)$$

The generalization error is then defined as:

$$G = I[f_n] - I_S[f_n]$$

Overfitting indicates that $I_S[f_n]​$ is small but $I[f_n]​$ is large, $f_n​$ will perform well on the training set but not perform well on other data from the joint probability distribution $p(x, y) ​$.

### KL-Divergence

Discrete case: $D_{KL}(P||Q) = -\sum_{x\in X}P(x)log\bigg(\frac{Q(x)}{P(x)}\bigg)$

Continuous case: $D_{KL}(P||Q) = \int_{-\infin}^{\infin} p(x)log\bigg(\frac{p(x)}{q(x)}\bigg) dx$

### Shannon Entropy

$H(X) = E\big[-log(P(X)\big]​$

### Bayesian vs. Frequentist

**Frequentist:** Sampling is infinite and decision rules can be sharp. Data are a repeatable random sample - there is a frequency. Underlying parameters are fixed i.e. they remain constant during this repeatable sampling process.

**Bayesian:** Unknown quantities are treated probabilistically and the state of the world can always be updated. Data are observed from the realized sample. Parameters are unknown and described probabilistically. It is the data which are fixed.

### Why big data works?



## Practical Procedures

### Deal with Missing Data

- Remove missing data rows
- Impute missing values: 1) use a constant value to distinguish missing for missing features, such as 0; 2) use a randomly sampled value based on this feature's distribution; 3) use a mean, median or mode value of this feature; 4) use a value predicted by another model

### Deal with Overfitting

Ways to detect: training/testing split

- Model-wise: 1) use regularization models: reduce variance by applying feature selection models, LASSO and Ridge (apply L1, L2 regularizer), random forest; 2) use k-fold cross validation; 3) apply ensemble models, Bagging, Boosting, a soft-max layer
- Data-wise: add more data;
- Deep learning: 1) early stopping; 2) drop-out 3) add regularizer for weights; 4) use data augmentation

### Identify Outliers

- Extreme value analysis: plot the histogram of individual features and exclude data points that is 3 standard deviation away if the data is Gaussian like. 
- Cluster based: use k-mans to cluster data. Exclude data points that are too far away from its centroid. 
- Use robust models such as LASSO, ridge and decision tree.

### Curse of Dimensionality

The distance measure increases as number of dimension grows and the feature space becomes sparse. The effects include: 1) the resulting lower data density requires more observations to keep the average distance between data points the same. In other words, supervised learning becomes more difficult because predictions for new samples are less likely to be based on learning from similar training features; 2) the variance increases as they get more opportunity to overfit to noise in more dimensions, resulting in poor generalization performance.

**Possible cause**: 1) high cardinality categorical variables would introduce numerous amount of one-hot encoding features; 2) too many features in the original space

To counter, general approach is to apply dimension reduction techniques such as PCA, autoencoder. For specific high cardinality categorical variables issue, various of encoding algorithms based on correlation of such categorical attributes to the target or class variables could be used: 1) supervised ratio, $v_i = p_i/t_i​$; 2) weight of evidence, $v_i=log \frac{p_i/p}{n_i/n}​$(better for imbalanced data).  

### Is the Coin Flipping Fair?

Suppose the coin is tossed 10 times and 8 heads are observed:

**P-value approach**: $H_0​$: null hypothesis, $p=0.5​$, $H_1​$: alternative hypothesis, $p > 0.5​$: the p-value is the probability of the observed outcome or something more extreme than the observed outcome, computed under the assumption that the null hypothesis is true (type 1 error). Under the fair assumption, $p=p(8 heads) + p(9 heads) + p(10 heads) = 0.055​$. If we define the "small" be $\alpha=0.05​$, which is smaller than p value, you would say 8 heads in 10 tosses is not enough evidence to conclude that the coin is not fair. The above mentioned is a one-tail test. You could also assume $H_1: p \neq 0.5​$ which you need to do a two tail test. 

If $H_1​$ is changed to $p=0.7​$, we can calculate the type II error. $p=1 - (p(8 heads) + p(9 heads) + p(10 heads) )=0.617​$

### Feature Selection

We need it to avoid **Multicollinearity**: one feature can be linearly predicted from the others with a substantial degree of accuracy resulting $X^T X​$ not invertible. To detect, 1) large changes in the estimated regression coefficients when a predictor variable is added or deleted; 2) insignificant regression coefficients for the affected variables in the multiple regression, but a rejection of the joint hypothesis that those coefficients are all zero (use F-test to give score for each feature, one feature may have very high F-score but its coefficient in the multiple regression model is small); 3) look at correlation between features. 

**F-test**: on regression setting it is used to determine individual feature importance as defined by $F_j=\frac{explained\;variance}{unexplained\;variance}= \frac{\frac{RSS_1-RSS_2}{p_2-p_1}}{\frac{RSS_1}{n-p_1}}$, where model 1 has only intercept as predictors and model 2 has the jth feature. F will have an F-distribution with $(p_2-p_1, n-p_2)$ degree of freedom.  he null hypothesis is rejected if the *F* calculated from the data is greater than the critical value of the *F*-distribution for some desired false-rejection probability (e.g. 0.05). In my kernel paper, I adopted a straight forward formula:

$\rho_j = \frac{(X_j-\bar{X_j})(Y-\bar{Y})}{\sqrt{var(X_j)var(Y)}}$

$F_j=\frac{2\rho_j}{1-2\rho_j}(N-1)$

### Model Selection



### Model Evaluation

Classification: confusion matrix, $precision=\frac{TP}{TP+FP}$, $recall=\frac{TP}{TP+FN}$, a measure that combines them is $F-score=2\times\frac{precision\times recall}{precision+recall}$

Regression: $RMSE=\sqrt{\frac{\sum_{t=1}^{T}(y_t-\hat{y_t})^2}{T}}$; $R^2$, it ranges from zero to one, with zero indicating that the proposed model does not improve prediction over the mean model, and one indicating perfect prediction. Improvement in the regression model results in proportional increases in R-squared.

#### A/B Test and Hypothesis Testing

In hypothesis testing, the null hypothesis is assumed to be true, and unless the test shows overwhelming evidence that the null hypothesis is not true, the null hypothesis is accepted.

|                             | $H_0$ is in fact true | $H_0$ is in fact false |
| --------------------------- | --------------------- | ---------------------- |
| Test decides $H_0$ is true  | Correct               | Type II error          |
| Test decides $H_0$ is false | Type I error          | Correct                |

P-value is type I error. Probability of a type I error can be held at some (preferably small level) while decreasing the probability of a type II error by increasing the sample size. 

#### Logistic Regression vs. SVMs

 The fundamental difference is that SVM minimizes hinge loss while logistic regression minimizes logistic loss which implies: 1) logistic loss diverges faster than hinge loss. So, in general, it will be more sensitive to outliers; 2) logistic loss does not go to zero even if the point is classified sufficiently confidently. This might lead to minor degradation in accuracy. Try logistic regression first and see how you do with that simpler model. If logistic regression fails and you have reason to believe your data won’t be linearly separable, try an SVM with a non-linear kernel like a Radial Basis Function (RBF).

## Mostly Used models

### Logistic Regression

$log(\frac{p(y=1)}{1-p(y=1)}) = \beta^{T}X​$ implies a sigmoid activation function, which can be seen as $p(y=1) = \frac{1}{1+e^{-\beta^TX}} \in (0, 1)​$ . The decisions boundary of it is 0.5. 

Loss function is logistic loss: $$argmin\sum_{i}L(y_i, f(x_i))​$$, where $L(y,f(x))=log\bigg(1+e^{-yf(x)}\bigg)​$

It is necessary to perform feature selection (remove uncorrelated features, filter out highly correlated features) since the model itself does not enforce it. It is possible to enforce regularizer by including it in the loss term such as $argmin\sum_{i}L(y_i, f(x_i)) + \lambda|w|^2​$.

Pros: very efficient for large amount of data solving by gradient descent;

Cons: need lots of data to scale well.

**Multi-class logistic regression**: applying a soft-max activation function as classifier would change binary logistic regression to multiclass. The loss function becomes cross-entropy loss.

### Decision Tree

Pro: easy to understand, have value even with small amount of data and is a white box;

Con: unstable, very high variance depending on training data. it is often inaccurate.

### Gradient Boosting (sequential)

**Compared to Ada-Boost**: a generalized version of Ada-Boost. The difference is that it does not have a particular loss function, any differentiable loss function works. Decision trees are used as weak learners. In Ada-Boost, only one stamp trees are used. In GB, more stamps tree is supported. 

The predictions of each tree are added together sequentially. After calculating the loss, to perform the gradient descent procedure, we must add a tree to the model that reduces the loss (i.e. follow the gradient). We do this by parameterizing the tree, then modify the parameters of the tree and move in the right direction by reducing the residual loss.The contribution of each tree to this sum can be weighted to slow down the learning by the algorithm. This weighting is called a shrinkage or a learning rate.

**Compared to logistic regression**: if all features are binary, they are equivalent with large dataset. A single gradient boosting decision stump is: $S=\sum_{i=1}^{n}(a_i-b_i)x_i+b_i$which is equal to $S=c_0+\sum_{i=1}^{n}c_ix_i$ , which is exactly the logit of logistic regression. If cross-entropy loss is used for GB, then they are the same if the number of stumps in GB is large enough. GB is robust against multilinearity though logistic regression doesn't.  

### Bagging (parallel)

Among all those training examples, randomly pick a subset from them with replacement and train decision tree for B times (usually called bootstrap aggregating). The final model is evaluated by averaging prediction of all B trees (regression) or majority vote (classification). 

### Random Forest

Same as Bagging except for each tree, the features are also randomly selected (typically $\sqrt{p}$ features to select) to avoid highly correlated trees in final pool to avoid high variance. Use cross-validation to determine number of trees. 

It can also be used to evaluate feature importance. To measure the importance of the j-th feature after training, the values of the j-th feature are permuted (randomly assigned) among the training data and the out-of-bag error is again computed on this perturbed data set. The importance score for the j-th feature is computed by averaging the difference in out-of-bag error before and after the permutation over all trees. The score is normalized by the standard deviation of these differences.

### Support Vector Machine

For linearly separable data, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The region bounded by these two hyperplanes is called the "margin", and the maximum-margin hyperplane is the hyperplane that lies halfway between them, geometrically as: $max \frac{2}{|w|}$

For non-separable data, use a soft-margin, namely hinge loss: $min \bigg[ \frac{1}{n}\sum_{i=1}^{n}max(0, 1-y_i(w^Tx_i-b))\bigg]+\lambda|w|^2$, parameter  $\lambda$  determines the trade-off between increasing the margin size an sure each data point lies at its correct region. Thus with a small enough $\lambda$ the model is similar to hard-margin SVM.

**large margin classifier**: the largest margin is found in order to avoid overfitting in SVM by maximizing the distance between neg. and pos. samples.

**$C$ in SVM**: regularization constant (inverse of $\lambda$ in ridge). For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.

**Use of Kernel**: SVM extends by using kernel tricks, transforming datasets into rich features space, so that complex problems can be still dealt with in the same “linear” fashion in the lifted hyper space. The time complexity of kernel methods scales with the number of training points, since you effectively learn a weight for each training point. SVMs have the nice mathematical property that most of those weights end up being zero, so you only have to keep around the examples with nonzero weights, i.e., the support vectors. That makes them more practical on larger datasets. 

**$\sigma$ in KSVM**: with lower $\sigma$ values, the model overfits, with higher $\sigma$ the model underfits. 

### Naïve Bayes Classifier

It is a probabilistic classifier based on applying Bayes' theorem and a strong assumption that all features are independent. 

A class's prior, $P(male)$, may be calculated by assuming equiprobable classes, or by calculating an estimate for the class probability from the training set (i.e., (prior for a given class) = (number of samples in the class) / (total number of samples)). The assumptions on distributions of features are called the event model of the Naive Bayes classifier: $P(X_1=x|male) = \frac{1}{\sqrt 2\pi \sigma^2} exp\bigg( \frac{-(x-\mu)^2}{2\sigma^2}\bigg)$

For example, $posterior(male|X) = \frac{P(male)\times P(X_1|male) \times P(X_2|male)}{P(X)}$

### Principle Component Analysis

Rotate the data to project the original feature into a new space where all features are orthogonal and features are ranked by maximum variance. $X_{n\times p}$, $(X^{T}X)_{p\times p}$, therefore the features can be at most p.

### Linear Discriminative Analysis



### Gaussian Mixture Model

GMM is tried to model the dataset as a mixture of several Gaussian distributions. Suppose there are K clusters, so $\mu$ and $\Sigma$ is also estimated for each k. Had it been only one distribution, they would have been estimated by **maximum-likelihood method**. But since there are K such clusters and the probability density is defined as a linear function of densities of all these K distributions: 

$p(X) = \sum_{k=1}^{K}\pi_{k}G(X|\mu_k,\Sigma_k)$, where $\pi_k$ is the mixing coefficient for k-th distribution.

To estimate the parameters by maximizing log-likelihood, $p(X|\mu, \Sigma, \pi)$:

$ln(p(X|\mu, \Sigma, \pi)) = \sum_{i=1}^{N}ln\sum_{k=1}^{K}\pi_kG(X_i|\mu_k,\Sigma_k)$

Taking derivative of the above respect to each parameter would give us the model, but there is no close form. So EM has to be used. 

### EM-Algorithm 

The Expectation-Maximization (EM) algorithm is an iterative way to find maximum-likelihood estimates for model parameters when the data is incomplete or has some missing data points or has some hidden variables. EM chooses some random values for the missing data points and estimates a new set of data. These new values are then recursively used to estimate a better first data, by filling up missing points, until the values get fixed.

**E step**: 1) initialize $\mu_k$, $\Sigma_k$ and $\pi_k$ by some random values, or by K means clustering results or by hierarchical clustering results; 2) then for those given parameter values, estimate the value of the latent variables such as $\gamma_k$.

**M step**: update the values of the parameters, $\mu_k$, $\Sigma_k$ and $\pi_k$ calculated by the derivative expressions in maximizing likelihood. 

**End criteria**: if the log-likelihood value converges to some value then stop.

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

An unsupervised classification density based clustering method. Points are classified as core points, density-reachable points and outliers.

1. A point p is a core point if at least minPts points are within distance ε (ε is the maximum radius of the neighborhood from p) of it (including p). Those points are said to be *directly reachable* from p;
2. A point q is directly reachable from p if point q is within distance ε from point p and p must be a core point;
3. A point q is reachable from p if there is a path *p*1, ..., *pn* with *p*1 = *p* and *pn* = *q*, where each pi+1 is directly reachable from pi (all the points on the path must be core points, with the possible exception of q);
4. All points not reachable from any other point are outliers.

A cluster then satisfies two properties: 1) all points within the cluster are mutually density-connected; 2) if a point is density-reachable from any point of the cluster, it is part of the cluster as well.

### Gaussian Process Regression





## Computer Vision

### Convolutional Neural Network

### AdaBoost for Face Recognition



## Natural Language Processing



## Statistics

### Binomial Distribution

$P(X=k)=\begin{pmatrix}n\\k\end{pmatrix}p^k(1-p)^{n-k}$, $mean=np$, $variance=np(1-p)$ 

### Bernoulli distribution

$P(X=1)=p$, $P(X=0)=1-p$, essentially a Binomial distribution with $n=1$

### Mean

Sample mean, $\overline{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$ for a random subset of entire population; population mean, $\mu = \frac{1}{n}\sum_{i=1}^{N}x_i$for entire population. 

### Simulation

**Multivariate Normal**: 

Let $$\sigma^2=\begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2\\\rho\sigma_1\sigma_2& \sigma_2^2 \end{bmatrix}$$, $\mu=\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}$, and $z=\begin{bmatrix}z_1\\z_2\end{bmatrix}$from sampling standard Normal distribution, then we have:

$M=chol(\sigma^2)^T$, $b=(M Z)^T + \mu^T$

**Multinomial**:

The probability mass function is: $f(x_1, x_2, ...x_k)=\frac{n!}{x_1!x_2!...x_k!}p_1^{x_1}p_2^{x_2}\times...p_k^{x_k}$

Various methods may be used to simulate a multinomial distribution. A very simple one is to use a random number generator to generate numbers between 0 and 1. First, we divide the interval from 0 to 1 in *k* subintervals equal in size to the probabilities of the *k* categories. Then, we generate a random number for each of n trials and use a **binary search** to classify the virtual measure or observation in one of the categories. 

## Time Series 



## Open Ended Questions

### How would you measure how much users liked videos?



### How to build a  news classifier for articles? 



### How does Facebook news feed work?









