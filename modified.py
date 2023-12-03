from part_a.utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

from grouping import *


def load_data(base_path="data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """

    train_path = os.path.join(base_path, "train_sparse.npz")
    train_matrix = load_train_sparse(train_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=50, b=10):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.layer1 = nn.Linear(num_question, k)
        self.layer2 = nn.Linear(k, b)
        self.dropout1 = nn.Dropout(0.7)
        self.layer3 = nn.Linear(b, k)
        self.dropout2 = nn.Dropout(0.7)
        self.layer4 = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        l1_w_norm = torch.norm(self.layer1.weight, 2) ** 2
        l2_w_norm = torch.norm(self.layer2.weight, 2) ** 2
        l3_w_norm = torch.norm(self.layer3.weight, 2) ** 2
        l4_w_norm = torch.norm(self.layer4.weight, 2) ** 2
        return l1_w_norm + l2_w_norm + l3_w_norm + l4_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # Added two layers with dropout                                     #
        #####################################################################
        f = self.layer1(inputs)
        g = F.sigmoid(f)
        h = self.layer2(g)
        i = self.dropout1(h)
        j = self.layer3(i)
        k = self.dropout2(j)
        l = self.layer4(k)
        out = F.sigmoid(l)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, data_dict, user_ids, plot=False):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param data_dict: Dict
    :param user_ids: List
    :param plot: bool
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # num_student = train_data.shape[0]
    

    # Storing objectives for plotting
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    # for train acc
    train_data_dict = {}
    if plot:
        # train_data_dict = load_train_csv("data")
        train_data_dict = data_dict
    

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in user_ids:
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (model.get_weight_norm() * lamb / 2)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        if plot:
            train_acc = evaluate(model, zero_train_data, train_data_dict)
            valid_acc = evaluate(model, zero_train_data, valid_data)
            valid_loss = compute_valid_loss(model, zero_train_data, valid_data)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accuracies.append(train_acc)
            valid_accuracies.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}".format(epoch, train_loss))

    if plot:
        plot_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)


def compute_valid_loss(model, train_data, valid_data):
    """ Evaluate the valid_data loss on the current model.

        :param model: Module
        :param train_data: 2D FloatTensor
        :param valid_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
        :return: float
        """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total_loss = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        target_label = valid_data['is_correct'][i]
        # index for output is valid_data['question_id'][i], since rows
        # don't match up
        prediction = output[0][valid_data['question_id'][i]].item()
        total_loss += (prediction - target_label) ** 2

    return total_loss


def plot_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    _, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].set_title('Training Loss over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].plot(train_losses, color='blue', label='Training Loss')

    axes[0, 1].set_title('Validation Loss over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].plot(valid_losses, color='orange', label='Validation Loss')

    axes[1, 0].set_title('Training Accuracy over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].plot(train_accuracies, color='blue', label='Validation Accuracy')

    axes[1, 1].set_title('Validation Accuracy over Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].plot(valid_accuracies, color='orange', label='Validation Accuracy')

    plt.tight_layout()
    plt.show()


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Overwrite valid_data and test_data with groups
    valid_data = generate_age_data("data/valid_data.csv", age=0)
    test_data = generate_age_data("data/test_data.csv", age=0)

    age_dic = group_user_id_by_age()
    data_dict = generate_age_data("data/train_data.csv", age=0)

    # gender_dic = group_user_id_by_gender()
    # data_dict = generate_gender_data("data/train_data.csv", gender=0)

    # Set model hyperparameters.
    num_questions = train_matrix.shape[1]
    k = 50
    b = 20
    model = AutoEncoder(num_questions, k, b)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 35
    lamb = 0.001

    train(model=model, 
          lr=lr, 
          lamb=lamb, 
          train_data=train_matrix, 
          zero_train_data=zero_train_matrix, 
          valid_data=valid_data, 
          num_epoch=num_epoch, 
          data_dict=data_dict,
          user_ids=age_dic["<2004"], # change this according to group
          plot=True,
          )

    test_accuracy = evaluate(model, train_data=zero_train_matrix, valid_data=test_data)
    print(f"Test Accuracy :", test_accuracy)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
