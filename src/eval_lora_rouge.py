# coding:utf-8
"""
@Filename: eval_lora_rouge.py
@Description: lora微调后的模型指标 rouge的评估
@Author: Li Zongyu
@Time: 2026/4/23 14:40
"""
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from peft import PeftModel
from src.lora_model_inference import predict
from src.utils.rouge import cal_rouge_score

# =================================需要修改的一些变量 Start =================================

# 原模型参数文件路径
model_path = '../model/qwen'
# lora的训练参数文件路径
lora_path = '../model/qwen-lora/checkpoint-1000'
# 使用的设备
device = "cuda:0"
# 测试集合数据文件路径
test_data_file_path = "../data/analysis_test.json"

# =================================需要修改的一些变量 End ===================================

# 1.加载模型和分词器
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, dtype=torch.bfloat16,
                                             trust_remote_code=True).eval()

# 加载lora权重,合并lora的参数和原模型参数
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 2. 加载测试集
with open(test_data_file_path, "r", encoding="utf-8") as fp:
    data_list = json.load(fp)


# 3. 计算并打印 Rouge指标
rouge_1_score = 0
rouge_2_score = 0
rouge_l_score = 0


for data in tqdm(data_list):
    # 大模型的预测答案
    llm_response = predict(model, tokenizer, data["<input>"])
    # 计算rouge指标
    rouge_score = cal_rouge_score(llm_response, data["<output>"])
    # 累加计算rouge-1的F1值
    rouge_1_score += rouge_score["rouge-1_f"]
    # 累加计算rouge-2的F1值
    rouge_2_score += rouge_score["rouge-2_f"]
    # 累加计算rouge-L的F1值
    rouge_l_score += rouge_score["rouge-l_f"]

print("Rouge-1 avg: ", rouge_1_score / len(data_list))
print("Rouge-2 avg: ", rouge_2_score / len(data_list))
print("Rouge-L avg: ", rouge_l_score / len(data_list))
