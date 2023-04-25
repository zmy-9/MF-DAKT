# Multi-Factors Aware Dual-Attentional Knowledge Tracing
CIKM'2021: Multi-Factors Aware Dual-Attentional Knowledge Tracing.
(Tensorflow implementation for MF-DAKT)

This is the code for the paper: [Multi-Factors Aware Dual-Attentional Knowledge Tracing](https://dl.acm.org/doi/10.1145/3459637.3482372)  

If you find this code is useful for your research, please cite as:
```
Moyu Zhang, Xinning Zhu, Chunhong Zhang, Yang Ji, Feng Pan and Changchuan Yin. 2021. Multi-Factors Aware Dual-Attentional Knowledge Tracing. In the Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM’21), November 1-5, 2021, QLD, Australia. ACM, New York, NY, USA, 2585-2597.
```

## Setups
* Python 3.6+
* Tensorflow 1.14.0
* Scikit-learn 0.21.3
* Numpy 1.17.2

## How to run model
### If you want to calculate relations of questions, you can do as below:
```
python3 prepare_relation.py
```
However, if you use the file of prepare_relation.py to generate relations of questions, you also need to modify code in pre_train.py where the original code load file of npz but prepare_relation.py will store relation matrix in txt format.
### If you want to pre-train question representations, you can do as below:
```
python3 pre_question.py
```
(Thanks to Mr.Sun for finding that we uploaded wrong loss function codes in pre_train.py, and we have updated the wrong component.)
### If you want to predict students' answer, you can do as below:
```
python3 main.py
```

(If you have any questions, please contact me on time. My E-mail is zhangmoyu@bupt.cn.)
