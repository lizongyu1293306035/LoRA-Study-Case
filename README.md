# LoRA微调案例——面向风险评估报告生成的LoRA微调
本案例旨在通过对`Qwen2.5-7B-Instruct`模型进行LoRA微调其风险报告生成能力，让读者了解LoRA微调的整个流程，包括LoRA训练过程以及LoRA微调过程中超参数的设置及其含义等。本案例提供了`.py`文件和`.ipynb`两种格式的文件，两类代码文件的代码一致，支持您直接使用`Python解释器`或`jupyter notebook` 中运行本示例代码。


## 快速开始

### 环境配置

下载并安装`peft`和`datasets`等第三方Python库

```shell
pip install -r requirements.txt
```

### 下载模型

本实例主要针对`Qwen2.5系列`模型进行 LoRA微调，故数据转化格式代码、LoRA超参数设置等均按照Qwen模型进行设置，若更换其他模型请更改其他模型的数据格式和LoRA超参数，以获得较好的微调效果。

国内的用户可以使用[ModelScope](https://www.modelscope.cn/)可以快速下载模型参数，国外的用户可以使用[Huggingface](https://huggingface.co/)下载。这里以国内为例在ModelScope上下载`Qwen2.5-7B-Instruct`模型参数文件。在终端运行以下命令即可开始下载：

```shell
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./model/qwen
```

其中修改传参`--local_dir 文件路径`可以下载到指定文件路径。这里我们默认下载到了项目目录下的`./model/qwen`文件夹下。

### 模型LoRA微调

运行`.\src\lora_finetuning.py`，修改代码中的一些变量，例如模型参数文件路径`model_file_path="你的Qwen模型参数文件路径"`，指定训练数据集等。运行以下命令即可开始训练，训练好的模型文件会保留在默认文件夹`./model/qwen-lora`下，故每次训练前请及时修改模型输出路径，防止模型参数文件覆盖掉其他文件。

```shell
python lora_finetuning.py
```

或

在`jupyter notebook`中直接运行`lora_finetuning.ipynb`即可开始训练。

### LoRA模型推理

运行`lora_model_inference.py`文件以开始模型推理，查看训练效果：

```shell
python lora_model_inference.py
```

或

在`jupyter notebook`中直接运行`lora_model_inference.ipynb`即可开始模型推理。

### LoRA模型指标评估

运行`eval_lora_rouge.py`文件以开始评估模型的`ROUGE`指标，查看微调前后的指标的提高：

```shell
python lora_model_inference.py
```

或

在`jupyter notebook`中直接运行`.ipynb`即可开始模型指标评估。



