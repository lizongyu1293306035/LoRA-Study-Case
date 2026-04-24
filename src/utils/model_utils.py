# coding:utf-8
"""
@Filename: model_utils.py
@Description: 对于模型的工具类
@Author: Li Zongyu
@Time: 2026/4/23 16:59
"""
import torch


# import bitsandbytes

def find_all_linear_names(model):
    # linear_classes = [torch.nn.Linear, bitsandbytes.nn.Linear4bit]
    linear_classes = [torch.nn.Linear]
    target_modules = set()

    for name, module in model.named_modules():
        if any([isinstance(module, cls) for cls in linear_classes]):
            parts = name.split('.')
            target_modules.add(parts[-1])

    return list(target_modules)


# print(find_all_linear_names())
