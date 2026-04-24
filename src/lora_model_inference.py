# coding:utf-8
"""
@Filename: lora_model_inference.py
@Description: 使用微调后的 LoRA模型进行模型的推理
@Author: Li Zongyu
@Time: 2026/4/23 14:40
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# =================================需要修改的一些变量 Start =================================

# 原模型参数文件路径
model_path = '../model/qwen'
# lora的训练参数文件路径
lora_path = '../model/qwen-lora/checkpoint-1000'
# 使用的设备
device = "cuda:0"

# =================================需要修改的一些变量 End ===================================


def predict(model, tokenizer, prompt: str) -> str:
    """
    模型回答

    :param model: 模型对象
    :param tokenizer: 分词器
    :param prompt: 问题
    :return: 模型的回答
    """
    inputs = tokenizer.apply_chat_template([{"role": "system", "content": "你是一个煤矿安全领域的专家，请帮我生成风险分析报告"},
                                            {"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           ).to(device)

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ans


def main():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    qwen_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, dtype=torch.bfloat16,
                                                      trust_remote_code=True).eval()

    # 加载lora权重,合并lora的参数和原模型参数
    lora_model = PeftModel.from_pretrained(qwen_model, model_id=lora_path)

    # 用户输入
    prompt = "2021年2月22日 内蒙古阿拉善新井煤业有限公司露天煤矿 边坡失稳发生特别重大坍塌事故。"

    # 模型问答
    print("LLM: " + predict(lora_model, tokenizer, prompt))


if __name__ == '__main__':
    main()
