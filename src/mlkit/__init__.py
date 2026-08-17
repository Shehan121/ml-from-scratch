"""Machine learning algorithms implemented from scratch with numpy.

Nothing here imports scikit-learn. It appears only in ``tests/``, as the reference
these implementations are verified against, and in ``scripts/`` to load the
bundled datasets.

    mlkit.metrics         accuracy, precision/recall/F1, confusion matrix, ROC-AUC, log loss
    mlkit.preprocessing   scalers, train/test split, K-fold, one-hot
    mlkit.linear          least squares three ways, ridge
    mlkit.logistic        binary logistic and softmax regression
    mlkit.knn             k-nearest neighbours
    mlkit.kmeans          k-means with k-means++
    mlkit.tree            CART decision tree
    mlkit.naive_bayes     Gaussian naive Bayes
    mlkit.pca             PCA via SVD
    mlkit.neural_net      layers, backpropagation, MLP classifier
    mlkit.optimizers      SGD, Momentum, Adam
    mlkit.gradcheck       finite-difference verification of the backprop maths
    mlkit.adversarial     FGSM and PGD against the from-scratch network
"""

__version__ = "1.0.0"
