# coding:utf-8
"""
@Filename: rouge.py
@Description: 计算rouge指标
@Author: Li Zongyu
@Time: 2026/4/24 14:20
"""
from rouge_chinese import Rouge
import jieba


def cal_rouge_score(predict: str, ground_truth: str) -> dict:
    """
    计算rouge值

    :param predict: 模型预测
    :param ground_truth: 真实答案
    :return: rouge-1、rouge-2、rouge-l
    """
    hypothesis_seg = ' '.join(jieba.cut(predict))
    reference_seg = ' '.join(jieba.cut(ground_truth))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis_seg, reference_seg)
    return {
        "rouge-1_r": scores[0]["rouge-1"]["r"],
        "rouge-1_p": scores[0]["rouge-1"]["p"],
        "rouge-1_f": scores[0]["rouge-1"]["f"],
        "rouge-2_r": scores[0]["rouge-2"]["r"],
        "rouge-2_p": scores[0]["rouge-2"]["p"],
        "rouge-2_f": scores[0]["rouge-2"]["f"],
        "rouge-l_r": scores[0]["rouge-l"]["r"],
        "rouge-l_p": scores[0]["rouge-l"]["p"],
        "rouge-l_f": scores[0]["rouge-l"]["f"],
    }


if __name__ == '__main__':
    print(cal_rouge_score("你好", "你好"))
