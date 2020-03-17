from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores = X.dot(W)
   
    # for each observation
    for obs in range(len(X)):
        numerator = 0.0
        denominator = 0.0
        
        # for each class
        for cls in range(scores.shape[1]):
            # if the true score
            if cls == y[obs]:
                numerator += np.exp(scores[obs,cls])

            denominator += np.exp(scores[obs,cls])
                                  
        loss += -1*np.log(numerator/denominator)
        
        # the gradient component for each class score for each observation will be
        # probability of that class for that observation mulitplied by the x_i
        # if class is true class, then subtract 1 from the probs (cls==y[obs]) is an
        # indicator function
        for cls in range(scores.shape[1]):
            probs = np.exp(scores[obs,cls])/denominator
            dW[:,cls] += (probs-(cls==y[obs])) * X[obs,:]
        
    # regularization and averaging
    loss /= len(X)
    loss += 0.5*reg*np.sum(W*W)
    
    dW /= len(X)
    dW += reg * W
                             
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    exp_scores = np.exp(scores)
    numerators = exp_scores[np.arange(len(exp_scores)), y] # the exp. score at the correct class
    denominators = np.sum(exp_scores, axis=1) # the sum of the exp. scores
    
    loss = np.sum(-np.log(numerators/denominators))
        
    probs = exp_scores/denominators.reshape((len(X),1))
    
    
    # the gradient component for each class score for each observation will be
    # probability of that class for that observation mulitplied by the x_i
    # if class is true class, then subtract 1 from the probs (cls==y[obs]) is an
    # indicator function
    
    # subtract 1 from the probability of the true class
    # now probs holds the factor by which we will add/subtract the corresponding x_i
    # from our gradient
    probs[np.arange(len(probs)), y] -= 1
    dW = (X.T).dot(probs)    
    
    # regularization and averaging
    loss /= len(X)
    loss += 0.5*reg*np.sum(W*W)
    
    dW /= len(X)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
