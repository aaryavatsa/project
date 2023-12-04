from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random


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
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score), end="\r")
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, neg_lld_train_list, neg_lld_val_list


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
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    iterations = 40

    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = \
        irt(train_data, val_data, lr, iterations)
    

    # Plot the training curve of training and validation log-likelihood as a function of iterations.
    plt.figure()
    plt.plot(range(iterations), train_lld_lst, label="train")
    plt.plot(range(iterations), val_lld_lst, label="validation")
    plt.xlabel("Iterations")
    plt.ylabel("Negative log-likelihood")
    plt.legend()
    plt.savefig("irt_training_curve.png")

    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)

    print("Validation accuracy: {}".format(val_acc))
    print("Test accuracy: {}".format(test_acc))

    # Part d
    # Select three questions j1,j2, and j3. Using the trained θ and β, plot three curves on the same plot 
    # that shows the probability of the correct response p(cij = 1) as a function of θ given a question j. 
    # Comment on the shape of the curves and briefly describe what these curves represent.

    j1 = 0
    j2 = 50
    j3 = 100

    theta_range = np.linspace(-3, 3, 100)
    p1 = sigmoid(theta_range - beta[j1])
    p2 = sigmoid(theta_range - beta[j2])
    p3 = sigmoid(theta_range - beta[j3])

    plt.figure()
    plt.plot(theta_range, p1, label=f"j1 = {j1}")
    plt.plot(theta_range, p2, label=f"j2 = {j2}")
    plt.plot(theta_range, p3, label=f"j3 = {j3}")
    plt.xlabel("Theta")
    plt.ylabel("Probability of correct response")
    plt.legend()
    plt.savefig("p4d.png")
    # N, D = 542, 1774
    # q_list = random.choices(np.arange(D), k=3)
    # plot_legends = []
    # for i in range(3):
    #     plot_legends.append(f'question_id: {q_list[i]}')

    # theta_range = np.linspace(-5, 5, 100)

    # curve_colors = ['r', 'g', 'b']
    # fig, ax = plt.subplots()

    # for i in range(3):
    #     q = q_list[i]
    #     # list of probabilities p(c_uq) for each theta given question q
    #     prob_list = sigmoid(theta_range-beta[q])
    #     # plot p(c_uq) as a function of theta given question q
    #     ax.plot(theta_range, prob_list, curve_colors[i])

    # ax.xaxis.set_label_text('theta')
    # ax.yaxis.set_label_text('p(c_ij)')
    # ax.set_title(
    #     'p(c_ij) as a function of theta given five different questions')
    # ax.legend(plot_legends)
    # plt.savefig('p4d.png')

    # theta = np.sort(theta)
    # plt.figure(2)
    # for j in range(5):
    #     beta_j = beta[j]
    #     cij = sigmoid(theta - beta_j)
    #     plt.plot(theta, cij, label="Question #"+str(j))
    # plt.title("Probability of Correct Response vs. Theta")
    # plt.ylabel("Probability")
    # plt.xlabel("Theta")
    # plt.legend()
    # plt.savefig("p4d.png")

    
if __name__ == "__main__":
    main()
