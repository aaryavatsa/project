from part_a.utils import *
from grouping import *
from scipy.sparse import save_npz


sparse_matrix = load_train_sparse()

group_by_age = group_user_id_by_age()
group_by_gender = group_user_id_by_gender()

# create sparse matrix for each group as training data
for age in group_by_age:
    user_ids = group_by_age[age]
    sparse_matrix_by_age = sparse_matrix[user_ids, :]
    save_npz("data/train_by_age_" + str(age) + ".npz", sparse_matrix_by_age)


for gender in group_by_gender:
    user_ids = group_by_gender[gender]
    sparse_matrix_by_gender = sparse_matrix[user_ids, :]
    save_npz("data/train_by_gender_" + str(gender) + ".npz", sparse_matrix_by_gender)
