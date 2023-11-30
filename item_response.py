from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector - representing the ability of each student
    :param beta: Vector - representing the difficulty of each question
    :return: float
    """

    log_lklihood = 0.
    
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = theta[u] - beta[q]
        if data["is_correct"][i] == 1:
            log_lklihood += x - np.log(1 + np.exp(x))
        else:
            log_lklihood += -np.log(1 + np.exp(x))

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    
    d_theta = np.zeros(theta.shape)
    d_beta = np.zeros(beta.shape)

    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = theta[u] - beta[q]
        d_theta[u] += lr * (data["is_correct"][i] - sigmoid(x))
        d_beta[q] -= lr * (data["is_correct"][i] - sigmoid(x))

    theta += d_theta
    beta += d_beta

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    
    theta = np.zeros(len(set(data["user_id"])))
    beta = np.zeros(len(set(data["question_id"])))


    train_acc_lst = []
    val_acc_lst = []
    neg_lld_train_list = []
    neg_lld_val_list = []


    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_train_list.append(neg_lld_train)
        
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        neg_lld_val_list.append(neg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, neg_lld_train_list, neg_lld_val_list, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    iterations = 40

    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = \
        irt(train_data, val_data, lr, iterations)
    iterations_lst = list(range(iterations))

    print(np.max(beta), np.min(beta), beta[0], beta[1], beta[-1])

    plt.plot(iterations_lst, train_lld_lst, "r-", label="training")
    plt.plot(iterations_lst, val_lld_lst, "b-", label="validation")
    plt.xlabel("Iterations")
    plt.ylabel("Negative likelihood")
    plt.legend()
    plt.show()

    plt.plot(iterations_lst, val_acc_lst)
    plt.xlabel("Iterations")
    plt.ylabel("Validation accuracy")
    plt.show()

    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation accuracy: {}".format(val_acc))
    print("Test accuracy: {}".format(test_acc))
    




if __name__ == "__main__":
    main()
