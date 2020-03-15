from builtins import range
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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            # if the correct class
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                
                # at the incorrect class, add x_i
                dW[:,j] += X[i]
                
                # at the correct class, subtract x_i
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # average over all training examples
    dW /= num_train
    dW = dW + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get scores
    scores = X.dot(W)
    
    # get scores of true classes
    true_class_scores = scores[np.arange(len(scores)), y].reshape((len(scores),1))
    
    # get score_wrong - score_true + 1
    score_diff_with_margin = scores - true_class_scores + 1 # 1 is the margin
    
    # true classes will have values of exactly 1, want to replace with 0 so don't
    # add to the loss
    score_diff_with_margin[score_diff_with_margin == 1] = 0
    margin = np.maximum(score_diff_with_margin, 0)
    loss = np.sum(margin) / X.shape[0]

    # margin is a NxC matrix that has the contribution to the multiclass SVM loss
    # for each observation for each class (0 or the score diff with margin)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # we are subtracting/adding the observation x_i based on the class scores
    # for this same individual x_i
    
    # for gradient, make indicator of whether or not class contributes to the
    # loss function (and gradient)
    margin[margin>0] = 1
    
    # for each observation, how many classes add to the loss/gradient
    valid_count = np.sum(margin, axis=1)
    
    # Subtract in correct class (-s_y)
    # for correct class for observation, count how many times we need to subtract
    # s_y from the gradient
    margin[np.arange(len(scores)),y ] -= valid_count
    
    
    # X is NxD and margin is NxC
    dW = (X.T).dot(margin)
    
    dW /= X.shape[0]

    # add regularization to gradient
    dW = dW + reg * 2 *W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
