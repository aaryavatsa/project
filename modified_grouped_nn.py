from part_b_utils import *

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=50, b=10, p=0.07):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define functions.
        self.layer1 = nn.Linear(num_question, k)
        self.layer2 = nn.Linear(k, b)
        self.dropout1 = nn.Dropout(p)
        self.layer3 = nn.Linear(b, k)
        self.dropout2 = nn.Dropout(p)
        self.layer4 = nn.Linear(k, num_question)
        self.dropout3 = nn.Dropout(p)
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
        f = self.layer1(inputs)
        g = F.sigmoid(f)
        h = self.layer2(g)
        i = self.dropout1(h)
        j = self.layer3(i)
        k = self.dropout2(j)
        l = self.layer4(k)
        m = self.dropout3(l)
        out = F.sigmoid(m)
        return out


def train(model, lr, lamb, num_epoch, train_data, zero_train_data, valid_data, train_data_dict, user_ids, plot=False):
    """ Train the neural network, where the objective also includes a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param num_epoch: int
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param train_data_dict: Dict
    :param user_ids: List
    :param plot: bool
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Storing objectives for plotting
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

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

    # all empty lists if plot=False
    return train_losses, train_accuracies, valid_losses, valid_accuracies


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


def main():
    """ Set hyperparameters and run the modified neural network (train and evaluate).
    If plot=True, return losses and accuracies for later use. Otherwise, return empty lists.
    """
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.
    num_questions = train_matrix.shape[1]

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 25
    lamb = 0.001

    # Set plotting.
    plot = False

    # Overwrite valid_data and test_data with groups
    valid_data = generate_age_data("data/valid_data.csv", age=0)
    test_data = generate_age_data("data/test_data.csv", age=0)
    age_dic = group_user_id_by_age()
    train_data_dict = generate_age_data("data/train_data.csv", age=0)

    model = AutoEncoder(num_questions, k=50)
    old_train_losses, old_train_accuracies, old_valid_losses, old_valid_accuracies \
        = train(model, lr, lamb, num_epoch, train_matrix, zero_train_matrix, valid_data, train_data_dict,
                age_dic["<2004"],   # change according to group
                plot)

    test_accuracy = evaluate(model, train_data=zero_train_matrix, valid_data=test_data)
    print(f"Test Accuracy for <2004:", test_accuracy)

    # Overwrite valid_data and test_data with groups
    valid_data = generate_age_data("data/valid_data.csv", age=1)
    test_data = generate_age_data("data/test_data.csv", age=1)
    age_dic = group_user_id_by_age()
    train_data_dict = generate_age_data("data/train_data.csv", age=1)

    model = AutoEncoder(num_questions, k=10)
    med_train_losses, med_train_accuracies, med_valid_losses, med_valid_accuracies \
        = train(model, lr, lamb, num_epoch, train_matrix, zero_train_matrix, valid_data, train_data_dict,
                age_dic["2004-2005"],  # change according to group
                plot)

    test_accuracy = evaluate(model, train_data=zero_train_matrix, valid_data=test_data)
    print(f"Test Accuracy for 2004-2005:", test_accuracy)

    # Overwrite valid_data and test_data with groups
    valid_data = generate_age_data("data/valid_data.csv", age=2)
    test_data = generate_age_data("data/test_data.csv", age=2)
    age_dic = group_user_id_by_age()
    train_data_dict = generate_age_data("data/train_data.csv", age=2)

    model = AutoEncoder(num_questions, k=10)
    young_train_losses, young_train_accuracies, young_valid_losses, young_valid_accuracies \
        = train(model, lr, lamb, num_epoch, train_matrix, zero_train_matrix, valid_data, train_data_dict,
                age_dic[">2005"],  # change according to group
                plot)

    test_accuracy = evaluate(model, train_data=zero_train_matrix, valid_data=test_data)
    print(f"Test Accuracy for >2005:", test_accuracy)

    if plot:
        plot_age_groups(old_train_losses, old_train_accuracies, old_valid_losses, old_valid_accuracies,
                        med_train_losses, med_train_accuracies, med_valid_losses, med_valid_accuracies,
                        young_train_losses, young_train_accuracies, young_valid_losses, young_valid_accuracies)


if __name__ == "__main__":
    main()
