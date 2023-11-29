from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (user-based): {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy (item-based): {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    # User-based
    accuracies = []
    for k in k_values:
        accuracies.append(knn_impute_by_user(sparse_matrix, val_data, k))
    max_accuracy = max(accuracies)
    max_index = accuracies.index(max_accuracy)
    k_star = k_values[max_index]
    plt.plot(k_values, accuracies)
    plt.title('Validation accuracy vs. k-values (user-based)')
    plt.xlabel('k')
    plt.ylabel('Validation accuracy')
    plt.show()

    print(f"k* with highest performance on validation data (user-based): {k_star}")
    nbrs = KNNImputer(n_neighbors=k_star)
    mat = nbrs.fit_transform(sparse_matrix)
    acc = sparse_matrix_evaluate(test_data, mat)
    print(f"Final test accuracy (user-based): {acc}")

    # Item-based
    accuracies = []
    for k in k_values:
        accuracies.append(knn_impute_by_item(sparse_matrix, val_data, k))
    max_accuracy = max(accuracies)
    max_index = accuracies.index(max_accuracy)
    k_star = k_values[max_index]
    plt.plot(k_values, accuracies)
    plt.title('Validation accuracy vs. k-values (item-based)')
    plt.xlabel('k')
    plt.ylabel('Validation accuracy')
    plt.show()

    print(f"k* with highest performance on validation data (item-based): {k_star}")
    nbrs = KNNImputer(n_neighbors=k_star)
    mat = nbrs.fit_transform(sparse_matrix.T)
    acc = sparse_matrix_evaluate(test_data, mat.T)
    print(f"Final test accuracy (item-based): {acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
