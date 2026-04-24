# coding:utf-8
"""
@Filename: eval_lora_rouge.py
@Description: lora微调后的模型指标 rouge的评估
@Author: Li Zongyu
@Time: 2026/4/23 14:40
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_file_path = r"P:\Desktop\LoRA_textbook_case_studies\model\qwen"

device = "cuda"


print(torch.cuda.is_available())
# 1. 加载模型及其分词器
# model = AutoModelForCausalLM.from_pretrained(model_file_path, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_file_path, trust_remote_code=True)

print(tokenizer)
