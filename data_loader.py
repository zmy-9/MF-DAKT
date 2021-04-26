import numpy as np
import pandas as pd
from scipy import sparse


class Data_loader(object):
    def __init__(self):
        self.dataset = pd.read_csv("assist2009/assist2009.csv", encoding="latin1", index_col=False)

    def data_load(self):
        dataset = self.dataset
        dataset = dataset.groupby("user_id").filter(lambda x: len(x) >= 3)  # retain logs of uses who have more than three

        dataset = dataset[~dataset["skill_id"].isnull()] # remove NaN skills in dataset

        dataset["item_id"] = np.unique(dataset["problem_id"], return_inverse=True)[1]  # Relabel user ID
        dataset["user_id"] = np.unique(dataset["user_id"], return_inverse=True)[1]  # Relabel problem ID
        dataset.reset_index(inplace=True, drop=True)

        listOfKC = []
        length = 0
        for kc_raw in dataset["skill_id"].unique():
            m = str(kc_raw).split('_')
            length = max(length, len(m))
            for elt in m:
                listOfKC.append(str(int(float(elt))))
        listOfKC = np.unique(listOfKC)  # Sort all skill IDs and delete duplicate ones
        # print("The max length of skill is: ", length)
        dict1_kc = {}
        for k, v in enumerate(listOfKC):  # 0 skill1, 1 skill2, ...
            dict1_kc[v] = k  # dict1_kc[skill1] = 0
        # print(dict1_kc)

        dataset = dataset[dataset.correct.isin([0, 1])]  # Remove potential continuous outcomes
        dataset['correct'] = dataset['correct'].astype(np.int32)  # Cast outcome as int32

        num_users = 1 + dataset['user_id'].max()  # record the number of users
        num_items = int(1 + dataset['item_id'].max())  # record the number of problems
        num_skills = len(listOfKC)  # record the number of skills

        # record the attempts of students
        Q_wins = np.zeros((len(dataset["user_id"].unique()), len(listOfKC)))  # Build Q-matrix，user * skill
        Q_fails = np.zeros((len(dataset["user_id"].unique()), len(listOfKC)))  # Build Q-matrix，user * skill

        # Build student-last answer matrix
        Q_last = np.zeros((len(dataset["user_id"].unique()), len(listOfKC)))  # shape: user * skill
        Q_last_valid = np.zeros((len(dataset["user_id"].unique()), len(listOfKC)))  # record whether students answered the corresponding skills, 0 denotes never exercise


        list_user, list_item, list_skill, list_skill_nums, list_wins, list_fails, list_wins_nums, list_fails_nums, target, difficulty = [], [], [], [], [], [], [], [], [], []
        list_last_attempt, list_last_nums = [], []
        for user, item_id, correct, skill_ids, diff in zip(dataset['user_id'], dataset['item_id'],
                                                           dataset['correct'],
                                                           dataset['skill_id'], dataset['difficulty']):
            sub_user = [user]
            sub_item = [item_id]
            target.append(float(correct))
            difficulty.append(diff)
            sub_skill, sub_skill_nums, sub_wins, sub_fails, sub_wins_nums, sub_fails_nums = [], [], [], [], [], []
            sub_last_attempt, sub_last_nums = [], []
            for skill in str(skill_ids).split('_'):
                sub_skill.append(dict1_kc[str(int(float(skill)))])
                sub_skill_nums.append(1)
                sub_wins.append(dict1_kc[str(int(float(skill)))])
                sub_fails.append(dict1_kc[str(int(float(skill)))])
                sub_wins_nums.append(Q_wins[user, dict1_kc[str(int(float(skill)))]])
                sub_fails_nums.append(Q_fails[user, dict1_kc[str(int(float(skill)))]])
                sub_last_nums.append(1)
                if Q_last_valid[user, dict1_kc[str(int(float(skill)))]]:
                    if Q_last[user, dict1_kc[str(int(float(skill)))]]:
                        sub_last_attempt.append(dict1_kc[str(int(float(skill)))])  # corresponding skill was answered correctly
                    else:
                        sub_last_attempt.append(dict1_kc[str(int(float(skill)))] + num_skills)  # corresponding skill was answered wrongly
                else:
                    sub_last_attempt.append(dict1_kc[str(int(float(skill)))] + 2 * num_skills)  # # corresponding skill never was answered
                if correct == 1:
                    Q_wins[user, dict1_kc[str(int(float(skill)))]] += 1
                    Q_last[user, dict1_kc[str(int(float(skill)))]] = 1
                else:
                    Q_fails[user, dict1_kc[str(int(float(skill)))]] += 1
                    Q_last[user, dict1_kc[str(int(float(skill)))]] = 0
                Q_last_valid[user, dict1_kc[str(int(float(skill)))]] = 1
            len_ = length - len(sub_wins)
            list_user.append(sub_user)
            list_item.append(sub_item)
            list_skill.append(sub_skill + [num_skills] * len_)
            list_skill_nums.append(sub_skill_nums + [0] * len_)
            list_wins.append(sub_wins + [num_skills] * len_)
            list_fails.append(sub_fails + [num_skills] * len_)
            list_wins_nums.append(sub_wins_nums + [0] * len_)
            list_fails_nums.append(sub_fails_nums + [0] * len_)
            list_last_attempt.append(sub_last_attempt + [3 * num_skills] * len_)
            list_last_nums.append(sub_last_nums + [0] * len_)
        return list_user, list_item, list_skill, list_skill_nums, list_wins, list_fails, list_wins_nums, list_fails_nums,\
               target, difficulty, num_users, num_items, num_skills, list_last_attempt, list_last_nums
