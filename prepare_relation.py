import numpy as np
import pandas as pd

df = pd.read_csv("ednet/ednet_difficulty.csv", encoding="latin1", index_col=False)

# Build q-matrix
listOfKC = []
length = 0
for kc_raw in df["new_sort_skill_id"].unique():
    m = str(kc_raw).split('-')
    length = max(length, len(m))
    for elt in m:
        listOfKC.append(str(int(float(elt))))
listOfKC = np.unique(listOfKC)  # delete duplicate concept id
print("The max length of skill is: ", length)
dict1_kc = {}
for k, v in enumerate(listOfKC):  # 0 skill1, 1 skill2, ...
    dict1_kc[v] = k  # dict1_kc[skill1] = 0

df = df[df.correct.isin([0, 1])]  # Remove potential continuous outcomes
df['correct'] = df['correct'].astype(np.int32)  # Cast outcome as int32

num_question = int(1 + df['item_id'].max())
num_concept = len(listOfKC)


def generate_question_concept_matrix(df, n_q, n_c):
    tmp = set()
    Q_matrix = np.zeros((n_q, n_c))
    for item, skills in zip(df['item_id'], df['new_sort_skill_id']):
        if item in tmp:
            continue
        for skill in str(skills).split('-'):
            Q_matrix[item][dict1_kc[str(int(float(skill)))]] = 1
    return Q_matrix  # num_questions * num_skills


q_c = generate_question_concept_matrix(df, num_question, num_concept)
qc_ = q_c[:, :]

qc_num = np.sum(q_c, axis=1)  # num_questions
print(qc_num)

# calculate min concept number between two questions
a = np.multiply(np.ones((num_question, num_question)), np.expand_dims(qc_num, 0))
b = np.multiply(np.ones((num_question, num_question)), np.expand_dims(qc_num, 1))
min_matrix = a <= b  # bool symbol
min_matrix_ = np.array(min_matrix, dtype=int)   # bool -> 1/0 symbol
min_matrix__ = np.multiply(min_matrix_, a) + np.multiply(1-min_matrix_, b)
relation_matrix = np.matmul(qc_, qc_.transpose()) / min_matrix__

np.savetxt("ednet/pro_pro.txt", relation_matrix)