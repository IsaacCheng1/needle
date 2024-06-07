"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    X = None
    y = None
    with gzip.open(image_filename) as f:
        images = f.read()

        magic_number = struct.unpack_from('>i', images)
        assert magic_number[0] == 2051

        num_examples = struct.unpack_from('>i', images, offset=4)
        num_examples = num_examples[0]

        num_rows = struct.unpack_from('>i', images, offset=8)
        num_rows = num_rows[0]
        num_cols = struct.unpack_from('>i', images, offset=12)
        num_cols = num_cols[0]
        assert num_rows == 28
        assert num_cols == 28

        fmt = '>%dB' % (num_examples * num_rows * num_cols)
        pixels = struct.unpack_from(fmt, images, offset=16)
        X = np.array(pixels, dtype=np.float32).reshape((num_examples, num_rows * num_cols))
        X = X / 255

    with gzip.open(label_filename) as f:
        labels = f.read()

        magic_number = struct.unpack_from('>i', labels)
        assert magic_number[0] == 2049

        num_examples = struct.unpack_from('>i', labels, offset=4)
        num_examples = num_examples[0]

        fmt = '>%dB' % num_examples
        targets = struct.unpack_from(fmt, labels, offset=8)
        y = np.array(targets, dtype=np.uint8)

    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    m, _ = Z.shape
    loss = ndl.log(ndl.exp(Z).sum(axes=(1,))) - (Z * y_one_hot).sum(axes=(1,))
    return ndl.summation(loss) / m

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    num_examples, input_dim = X.shape
    _, num_classes = W2.shape
    for i in range(0, num_examples, batch):
        # sample 1 batch
        X_minibatch = X[i: i+batch] # (B,n)
        y_minibatch = y[i: i+batch] # (B,)
        y_one_hot = np.zeros((batch, num_classes)) # (B,K)
        y_one_hot[np.arange(batch), y_minibatch] = 1

        X_tensor = ndl.Tensor(X_minibatch)
        y_tensor = ndl.Tensor(y_one_hot)

        # forward pass (construct computational graph)
        Z = ndl.relu(X_tensor @ W1) @ W2 # (B,K)
        loss = softmax_loss(Z, y_tensor) # scalar

        # compute gradient
        loss.backward()

        # SGD
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return (W1, W2)

def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    n, k = X_tr.shape[1], max(y_tr) + 1
    np.random.seed(0)

    # initialize the weight
    W1 = ndl.Tensor(np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim))
    W2 = ndl.Tensor(np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k))

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        W1, W2 = nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)

        Z_tr = ndl.relu(ndl.Tensor(X_tr) @ ndl.Tensor(W1)) @ ndl.Tensor(W2)
        train_loss, train_err = loss_err(Z_tr, y_tr)

        Z_te = ndl.relu(ndl.Tensor(X_te) @ ndl.Tensor(W1)) @ ndl.Tensor(W2)
        test_loss, test_err = loss_err(Z_te, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))

if __name__ == "__main__":
    # load data
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    # train
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)