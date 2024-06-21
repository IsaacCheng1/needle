import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    module = nn.Residual(
        nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            norm(dim=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=hidden_dim, out_features=dim),
            norm(dim=dim)
        )
    )

    return nn.Sequential(
        module,
        nn.ReLU()
    )

def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
   modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]

   for _ in range(num_blocks):
       modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))

   modules.append(nn.Linear(hidden_dim, num_classes))
   return nn.Sequential(
       *modules
   )

def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    sum_of_loss = 0

    sum_of_hit = 0
    sum_of_examples = 0

    total_num_batches = 0

    loss_function = nn.SoftmaxLoss()

    if opt is not None:
        model.train()

        for i, (X, y) in enumerate(dataloader):
            total_num_batches += 1

            # forward pass
            logits = model(X)

            # reset grad
            opt.reset_grad()

            # compute loss
            loss = loss_function(logits, y)

            # update sum_of_loss
            sum_of_loss += loss.numpy()

            # update sum_of_hit and sum_of_examples
            sum_of_hit += (y.numpy() == np.argmax(logits.numpy(), axis=1)).sum()
            sum_of_examples += y.shape[0]

            # compute grad
            loss.backward()

            # update parameters
            opt.step()

    else:
        model.eval()

        for i, (X, y) in enumerate(dataloader):
            total_num_batches += 1

            # forward pass
            logits = model(X)

            # compute loss
            loss = loss_function(logits, y)

            # update sum_of_loss
            sum_of_loss += loss.numpy()

            # update sum_of_hit and sum_of_examples
            sum_of_hit += (y.numpy() == np.argmax(logits.numpy(), axis=1)).sum()
            sum_of_examples += y.shape[0]

    average_error_rate = (sum_of_examples - sum_of_hit) / sum_of_examples
    average_loss = sum_of_loss / total_num_batches

    return average_error_rate, average_loss


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)

    # initialize the train data loader
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)

    # initialize the test data loader
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # NN
    resNet = MLPResNet(28 * 28,
                       hidden_dim=hidden_dim)

    # initialize optimizer
    opt = optimizer(resNet.parameters(),
                    lr=lr, weight_decay=weight_decay)

    # train
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")

    for i in range(epochs):
        training_accuracy, training_loss = epoch(train_dataloader, resNet, opt)
        test_accuracy, test_loss = epoch(test_dataloader, resNet, None)

        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(i, training_loss, training_accuracy, test_loss, test_accuracy))

    return training_accuracy, training_loss, test_accuracy, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="./data")
