# coding:utf-8
"""
@Filename: data_utils.py
@Description: 数据格式转换的工具脚本
@Author: Li Zongyu
@Time: 2026/4/23 15:24
"""
import json
from datasets import Dataset
from transformers import AutoTokenizer


def load_dataset(system_prompt: str, data_file_path: str, tokenizer: AutoTokenizer = None) -> Dataset:
    """
    从json文件中读取数据

    :param system_prompt: 构建数据集时给定的系统提示词
    :param data_file_path: json文件的路径
    :param tokenizer: 模型的分词器，可选传参
    :return: Huggingface的 Dataset对象
    """
    # 1. 手动读取 JSON 文件
    with open(data_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建符合格式的数据集
    data_format = []

    for d in data["conversations"]:
        system_content = ""
        # 用户的问题
        user_content = d[0]["value"]
        # llm的回答
        assistant_content = d[1]["value"]
        if system_prompt:
            # 系统提示词
            system_content = system_prompt
            data_format.append(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ])
        else:
            data_format.append(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ])
    # 将数据转换为Hugging Face Dataset格式
    dataset = Dataset.from_dict({
        'conversations': data_format
    })

    return dataset
