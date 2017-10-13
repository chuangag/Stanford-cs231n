import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:, y[i]] += -X[i]
                dW[:, j] += X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]  # C
    num_train = X.shape[0]  # N
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    y_pred_score = X.dot(W)  # y_pred_score shape=(N,C)
    #print y_pred_score[:5]
    # making a one-hot matrix of correct class (correct class=1, other=0)
    y_correct = np.zeros((num_train, num_classes))
    y_correct[np.arange(num_train), y] = 1
    #print y_correct[:5]
    y_correct_score = y_pred_score*y_correct
    y_correct_score=y_correct_score[np.arange(num_train),y]
    #print y_correct_score[:5]
    margin_mat = y_pred_score - np.repeat(np.array([y_correct_score]).transpose(),num_classes,axis=1) + 1 # remove 1 form the correct class
    margin_mat[margin_mat < 0] = 0  # mask negative to 0
    margin_mat[np.arange(num_train),y]=0 # ignore for correct class
    loss = float(margin_mat.sum())/num_train
    loss += reg*np.sum(W*W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    margin_mask=margin_mat # (N,C)
    margin_mask[margin_mask>0]=1
    margin_mask[margin_mask<0]=0
    margin_mask[np.arange(num_train),y]=-1*np.sum(margin_mask,axis=1)
    dW=np.dot(X.T,margin_mask)
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW