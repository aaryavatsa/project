# TODO: complete this file.

from part_a.knn import *
from utils import *


def generate_sample(matrix):
    n = matrix.shape[0]
    point = np.random.choice(n, n, replace=True)
    sample = matrix[point]

    return sample


def find_best_k(sample, val_data):
    k_values = [1, 6, 11, 16, 21, 26]
    accuracies = []

    for k in k_values:
        accuracies.append(knn_impute_by_user(sample, val_data, k))
    max_accuracy = max(accuracies)
    max_index = accuracies.index(max_accuracy)
    k_star = k_values[max_index]

    return k_star


def knn_ensemble(matrix, val_data, test_data):
    val_accuracies = []
    test_accuracies = []
    for i in range(3):
        print(f"Prediction number {i}:")
        sample = generate_sample(matrix)
        k_star = find_best_k(sample, val_data)
        val_accuracy = knn_impute_by_user(sample, val_data, k_star)
        val_accuracies.append(val_accuracy)
        test_accuracy = knn_impute_by_user(sample, test_data, k_star)
        test_accuracies.append(test_accuracy)

    return val_accuracies, test_accuracies


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    val_accuracies, test_accuracies = knn_ensemble(sparse_matrix, val_data, test_data)
    final_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    final_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    print(f"Final validation accuracy: {final_val_accuracy}")
    print(f"Final test accuracy: {final_test_accuracy}")


if __name__ == "__main__":
    main()
