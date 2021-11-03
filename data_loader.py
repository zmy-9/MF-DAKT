import numpy as np
import pandas as pd
from scipy import sparse


class Data_loader(object):
    def __init__(self):
        self.df = pd.read_csv("ednet/ednet_difficulty.csv", encoding="latin1", index_col=False)

    def data_load(self):
        df = self.df
        '''
        # Filter out users that have less than min_interactions interactions
        df = df.groupby("user_id").filter(lambda x: len(x) >= 3)

        # Remove NaN skills
        df = df[~df["skill_id"].isnull()]

        df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
        df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
        df.reset_index(inplace=True, drop=True)
        '''
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

        num_student = 1 + df['user_id'].max()  # user/item/skill IDs are distinct
        num_question = int(1 + df['item_id'].max())
        num_concept = len(listOfKC)

        # Build Q-matrix
        Q_success = np.zeros((len(df["user_id"].unique()), len(listOfKC)))  # construct a Q-matrix，num_student*num_concepts, records students' successful records
        Q_fails = np.zeros((len(df["user_id"].unique()), len(listOfKC)))  # construct a Q-matrix，num_student*num_concepts, records students' failed records
        Q_step = np.zeros(len(df["user_id"].unique()))  # construct a Q-matrix，num_student*1, records the total timesteps of students

        # Build student-last practice matrix
        Q_last = np.zeros((len(df["user_id"].unique()), len(listOfKC)))  # construct a Q-matrix，recording the answer of students answered concept last time
        Q_last_valid = np.zeros((len(df["user_id"].unique()), len(listOfKC)))   # construct a Q-matrix，recording whether students ever answered concept last time
        Q_last_step = np.zeros((len(df["user_id"].unique()), len(listOfKC)))  # construct a Q-matrix，num_student*num_concepts, records the timestep of students practicing each concept last time

        list_user, list_question, list_concept, list_concept_nums, list_success, list_fails, list_success_nums, list_fails_nums, target, difficulty = [], [], [], [], [], [], [], [], [], []
        list_recent_attempt, list_recent_nums, list_recent_interval = [], [], []

        dic = {}  # record each students' records
        for user, item_id, correct, skill_ids, diff in zip(df['user_id'], df['item_id'],
                                                           df['correct'],
                                                           df['new_sort_skill_id'], df['difficulty']):
            sub_user = [user]
            sub_item = [item_id]
            target.append(float(correct))
            difficulty.append(diff)
            sub_skill, sub_skill_nums, sub_wins, sub_fails, sub_wins_nums, sub_fails_nums = [], [], [], [], [], []
            sub_last_attempt, sub_last_nums = [], []
            sub_last_interval = []
            cur_step = Q_step[user]
            for skill in str(skill_ids).split('-'):
                sub_skill.append(dict1_kc[str(int(float(skill)))])
                sub_skill_nums.append(1)
                sub_wins.append(dict1_kc[str(int(float(skill)))])
                sub_fails.append(dict1_kc[str(int(float(skill)))])
                sub_wins_nums.append(Q_success[user, dict1_kc[str(int(float(skill)))]])
                sub_fails_nums.append(Q_fails[user, dict1_kc[str(int(float(skill)))]])
                sub_last_nums.append(1)
                if Q_last_valid[user, dict1_kc[str(int(float(skill)))]]:
                    tmp_t = cur_step - Q_last_step[user, dict1_kc[str(int(float(skill)))]]
                    sub_last_interval.append(tmp_t)
                    if Q_last[user, dict1_kc[str(int(float(skill)))]]:
                        sub_last_attempt.append(dict1_kc[str(int(float(skill)))])  # 回答相关知识点正确
                    else:
                        sub_last_attempt.append(dict1_kc[str(int(float(skill)))] + num_concept)  # 回答相关知识点失败
                else:
                    sub_last_attempt.append(dict1_kc[str(int(float(skill)))] + 2 * num_concept)  # 从未回答过相关知识点
                    sub_last_interval.append(0)
                if correct == 1:
                    Q_success[user, dict1_kc[str(int(float(skill)))]] += 1
                    Q_last[user, dict1_kc[str(int(float(skill)))]] = 1
                else:
                    Q_fails[user, dict1_kc[str(int(float(skill)))]] += 1
                    Q_last[user, dict1_kc[str(int(float(skill)))]] = 0
                Q_last_valid[user, dict1_kc[str(int(float(skill)))]] = 1
                Q_last_step[user, dict1_kc[str(int(float(skill)))]] = cur_step
            Q_step[user] += 1
            len_ = length - len(sub_wins)
            list_user.append(sub_user)
            list_question.append(sub_item)
            list_concept.append(sub_skill + [num_concept] * len_)
            list_concept_nums.append(sub_skill_nums + [0] * len_)
            list_success.append(sub_wins + [num_concept] * len_)
            list_fails.append(sub_fails + [num_concept] * len_)
            list_success_nums.append(sub_wins_nums + [0] * len_)
            list_fails_nums.append(sub_fails_nums + [0] * len_)
            list_recent_attempt.append(sub_last_attempt + [3 * num_concept] * len_)
            list_recent_nums.append(sub_last_nums + [0] * len_)
            list_recent_interval.append(sub_last_interval + [0] * len_)
            if user not in dic:
                dic[user] = []
            dic[user].append([sub_user, sub_item, sub_skill + [num_concept] * len_, sub_skill_nums + [0] * len_,
            sub_wins + [num_concept] * len_, sub_fails + [num_concept] * len_, sub_wins_nums + [0] * len_,
                              sub_fails_nums + [0] * len_, sub_last_attempt + [3 * num_concept] * len_,
                              sub_last_nums + [0] * len_, sub_last_interval + [0] * len_, float(correct), diff])

        return dic, num_student, num_question, num_concept, length
