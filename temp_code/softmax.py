import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_class=W.shape[1]
  def softmax(x):
    shift_x=x-np.max(x)
    expx=np.exp(shift_x)
    return expx/np.sum(expx)

  y_pred=X.dot(W)
  for i in xrange(num_train):
    correct_class=y[i]
    loss+=-np.log(softmax(y_pred[i])[correct_class])
    y_correct=np.zeros(num_class)
    y_correct[correct_class]=1
    p=softmax(y_pred[i])
    for j in xrange(num_class):
      if j==correct_class:
        dW[:,j]+=p[j]*X[i]-X[i]
      else:
        dW[:,j]+=p[j]*X[i]
  loss=float(loss)/num_train+reg*np.sum(W*W)
  dW=dW/float(num_train)+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_class=W.shape[1]

  score=X.dot(W) # (N,C)
  shifted_score=score-np.repeat(np.array([np.max(score,axis=1)]).transpose(),num_class,axis=1) # (N,C)
  exp_score=np.exp(shifted_score) # (N,C)
  denom=np.sum(exp_score,axis=1) # (N)

  loss=np.sum(-np.log(exp_score[np.arange(num_train),y]/denom))
  loss=loss/float(num_train)+reg*np.sum(W*W)
  
  grad=exp_score/np.repeat(np.array([denom]).transpose(),num_class,axis=1) # (N,C)
  grad[np.arange(num_train),y]+= -1
  dW=np.dot(X.T,grad) # (D,N)*(N,C)=(D,C)
  dW=dW/float(num_train)+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

