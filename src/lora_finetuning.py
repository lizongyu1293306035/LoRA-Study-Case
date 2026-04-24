# coding:utf-8
"""
@Filename: lora_finetuning.py
@Description: lora微调
@Author: Li Zongyu
@Time: 2026/4/23 14:39
"""
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from src.utils.data_utils import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

print("显卡是否可用：" + str(torch.cuda.is_available()))

# =================================需要修改的一些变量 Start =================================

# Qwen模型的参数文件路径
model_file_path = r"..\model\qwen"

# 设置模型加载的设备，cuda:0 为使用第一个显卡资源。device="cpu" 为调用CPU资源进行训练
device = "cuda:0"

# 系统提示词，根据训练任务的不同需要有不同的系统提示词
system_prompt = "你是一个煤矿安全领域的专家，请帮我生成风险分析报告"

# 训练数据的文件路径
data_file_path = r"..\data\analysis_data.json"

# 每条训练数据的最大长度，分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
MAX_LENGTH = 2048

# LoRA微调后的模型文件输出路径
output_dir = r"..\model\qwen-lora"

# =================================需要修改的一些变量 End ===================================


# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_file_path, trust_remote_code=True)

# 2. 加载训练集
train_dataset = load_dataset(system_prompt=system_prompt,
                             data_file_path=data_file_path,
                             tokenizer=tokenizer)


def process_func(example: Dict[str, List]):
    """
    遍历dataset对象，将每条数据进行按照Qwen的输入格式整理、分词、掩码等操作

    :param example: 每条数据的system、user、assistant文本
    :return: 转化、编码后的数据
    """
    example = example["conversations"]

    instruction = tokenizer(
        f"<|im_start|>system\n{example[0]['content']}<|im_end|>\n<|im_start|>user\n{example[1]['content']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example[2]['content']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 编码后的数据集
train_dataset_tokenized = train_dataset.map(process_func, remove_columns=train_dataset.column_names)

print(tokenizer.decode(train_dataset_tokenized[0]['input_ids']))
print(tokenizer.decode(list(filter(lambda x: x != -100, train_dataset_tokenized[1]["labels"]))))

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(model_file_path, device_map=device)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 4. 配置LoRA 超参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 指定任务类型
    r=32,  # LoRa秩（秩越大，微调容量越大，同时资源消耗也更多）
    lora_alpha=16,  # 用于控制LoRA 更新模型参数的强度
    inference_mode=False,  # 训练模式
    target_modules=[
        "q_proj",
        "v_proj",
    ],  # 目标层
    lora_dropout=0.1,  # 丢弃率
    bias="none",  # 偏置项
)
model = get_peft_model(model, config)
# 打印LoRA 更新的参数量
model.print_trainable_parameters()

# 5.配置训练参数并开始训练
args = TrainingArguments(
    output_dir=output_dir,  # 训练后的模型参数保存文件路径
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    num_train_epochs=8,  # 改这里
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_steps=100,
    report_to="none",
)

# 开始训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset_tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
