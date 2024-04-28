import json
import os
import random
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
# %% 定义数据集
# from datasets import Dataset, load_dataset
from torch.utils.data import Dataset

train_inputs_ids = torch.randint(0, 10, (500,), dtype=torch.long)
train_outputs_ids = (train_inputs_ids + 3) % 10


eval_inputs_ids = torch.randint(0, 10, (1000,), dtype=torch.long)
eval_outputs_ids = (eval_inputs_ids + 3) % 10


class MyDataset(Dataset):
    def __init__(self, inputs_ids, outputs_ids):
        self.inputs_ids = inputs_ids
        self.outputs_ids = outputs_ids

    def __len__(self):
        return len(self.inputs_ids)

    def __getitem__(self, index):
        return self.inputs_ids[index], self.outputs_ids[index]


train_dataset = MyDataset(train_inputs_ids, train_outputs_ids)
eval_dataset = MyDataset(eval_inputs_ids, eval_outputs_ids)


print(train_dataset[0])
print(eval_dataset[0])

# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=8, shuffle=True)
# eval_dataloader = torch.utils.data.DataLoader(
#     eval_dataset, batch_size=8, shuffle=True)


def data_collator(fetures):
    input_ids, labels = fetures
    return {"input_ids": input_ids, "labels": labels}


# for batch in train_dataloader:
#     print(batch)
#     print(data_collator(batch))
#     break

# %% 定义模型
# 词向量编码的维度
embedding_size = 32
# 假设的类别数
num_classes = 10
# 隐藏层的特征数
hidden_size = 20


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_classes):
        super(TwoLayerNet, self).__init__()
        # 第一个线性层，从输入层到隐藏层
        self.embeding = nn.Embedding(input_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        # 第二个线性层，从隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embeding(x)
        # 通过ReLU激活函数传递第一个线性层的输出
        x = F.relu(self.fc1(x))
        # 通过第二个线性层得到输出
        x = self.fc2(x)
        return x


# 因为输入是一个特征，所以input_size是10
input_size = 10
model = TwoLayerNet(input_size, embedding_size, hidden_size, num_classes)


# 模型前向传播测试
# for batch in train_dataloader:
#     print(batch)
#     batch = data_collator(batch)
#     print(batch)
#     output = model(batch["input_ids"])
#     print(output)
#     break

# %% 定义训练参数

class TrainingConfig:
    def __init__(self, **kwargs):
        # 初始化训练参数
        self.learning_rate: float = 1e-3
        self.batch_size: int = 8
        self.gradient_accumulation_steps: int = 4
        self.train_dataloader_shuffle: bool = True
        self.eval_dataloader_shuffle: bool = False

        self.epochs: int = 10
        self.model_save_dir: str = r'D://code//python//outputtest//'
        self.device: str = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.step_log: int = 10
        self.global_step: int = 0
        self.global_epoch: int = 0
        self.save_epochs: int = 5
        self.save_step: int = 50

        # self.resume_from_checkpoint = r'D://code//python//outputtest//checkpoint_150//'
        self.resume_from_checkpoint = None

        # self.random_seed = torch.get_rng_state()
        self.random_seed: int = 233

        self.best_eval_indicator: float = float('-inf')

        # 使用 kwargs 来覆盖默认值（如果存在的话）
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr

    def set_batch_size(self, bs: int):
        self.batch_size = bs

    def set_epochs(self, epochs: int):
        self.epochs = epochs

    def set_model_save_dir(self, path: str):
        self.model_save_dir = path

    def set_device(self, device: str):
        self.device = device

    def set_train_dataloader_shuffle(self, shuffle: bool):
        self.train_dataloader_shuffle = shuffle

    def set_eval_dataloader_shuffle(self, shuffle: bool):
        self.eval_dataloader_shuffle = shuffle

    def set_step_log(self, step: int):
        self.step_log = step

    def set_save_epochs(self, epochs: int):
        self.save_epochs = epochs

    def set_random_seed(self, seed: int):
        self.random_seed = seed
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

    def add_global_step(self):
        self.global_step += 1

    def add_global_epoch(self):
        self.global_epoch += 1

    def set_gradient_accumulation_steps(self, gradient_accumulation_steps: int):
        self.gradient_accumulation_steps = gradient_accumulation_steps
    
    def set_save_step(self, save_step: int):
        self.save_step = save_step
    
    def set_best_eval_indicator(self, best_eval_indicator: float):
        self.best_eval_indicator = best_eval_indicator

    def to_dict(self):
        # 转换为字典，但不包括 device，因为它会在加载时重新计算
        config_dict = {
            key: getattr(self, key)
            for key in vars(self).keys() if key != 'device'
        }
        return config_dict

    def __str__(self):
        # 提供一个字符串表示，方便打印配置信息
        print_text = ''
        for key, value in self.__dict__.items():
            print_text += f"{key}: {value}\n"
        return f"TrainingConfig({print_text})"


config = TrainingConfig()
print(config)
print(config.__dict__)

# %% 定义Tensorboard的接口
from torch.utils.tensorboard import SummaryWriter

# %% 定义Dataloader

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=config.train_dataloader_shuffle)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, shuffle=config.train_dataloader_shuffle)


# %% 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# %% 接续训练
global resume_from_checkpoint_flag
# resume_from_checkpoint = None
if config.resume_from_checkpoint is not None:
    # 读取本地模型和配置
    resume_from_checkpoint = config.resume_from_checkpoint
    save_dict_resume = torch.load(os.path.join(resume_from_checkpoint, "model.pth"))
    config = TrainingConfig(**save_dict_resume['config'])
    print(config)
    model.load_state_dict(save_dict_resume["model"])
    model.to(config.device)
    optimizer.load_state_dict(save_dict_resume["optimizer"])
    scheduler.load_state_dict(save_dict_resume["scheduler"])
    
    resume_from_checkpoint_flag = True
    start_epoch = config.global_epoch
    # 设置 tqdm 的初始值  
    initial = config.global_step if config.global_step is not None else 0  
    # pbar = tqdm(total=config.epochs*len(train_dataloader) - initial, initial=initial, desc='Training')
    pbar = tqdm(total=config.epochs*len(train_dataloader), initial=initial, desc='Training')
else:
    resume_from_checkpoint_flag = False
    start_epoch = config.global_epoch
    # 设置 tqdm 的初始值  
    initial = config.global_step if config.global_step is not None else 0  
    # pbar = tqdm(total=config.epochs*len(train_dataloader) - initial, initial=initial, desc='Training') 
    pbar = tqdm(total=config.epochs*len(train_dataloader), initial=initial, desc='Training') 
  



# %%保存模型与配置文件的独立函数
def save_dict_and_config(save_dict: dict, config_dict: dict, checkpoint_dir: str, checkpoint_file: str):
    if(type(config_dict) == TrainingConfig):
        config_dict = config_dict.to_dict()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 保存模型
    torch.save(save_dict, os.path.join(checkpoint_dir, checkpoint_file))
    print("\nModel saved.")
    # 保存配置文件
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)
        print("\nconfig saved.")
        print("--------------------------------------------------------------------")


# %% 定义训练过程中的保存函数
def save_model_intrain(model: nn.Module, config: TrainingConfig, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler) -> None:
    steps = config.global_step
    model_save_dir = config.model_save_dir
    checkpoint_dir = os.path.join(model_save_dir, f'checkpoint_{steps}/')  
    checkpoint_file = f'model.pth'

    # 确保checkpoint目录存在  
    if not os.path.exists(checkpoint_dir):  
        os.makedirs(checkpoint_dir)
    # 获取model_save目录下所有文件夹  
    checkpoint_dirs = [d for d in os.listdir(model_save_dir) if os.path.isdir(os.path.join(model_save_dir, d))]  
      
    # 过滤出与checkpoint相关的文件夹，并按名称排序（这将基于steps的值）  
    checkpoint_dirs = sorted([d for d in checkpoint_dirs if d.startswith('checkpoint_')], key=lambda x: int(x.split('_')[1]))  
  
    # 如果文件夹数量超过3个，删除最老的checkpoint  
    if len(checkpoint_dirs) > 3:  
        oldest_dir = os.path.join(model_save_dir, checkpoint_dirs[0])  
        for file in os.listdir(oldest_dir):  
            file_path = os.path.join(oldest_dir, file)  
            if os.path.isfile(file_path):  
                os.unlink(file_path)  
        os.rmdir(oldest_dir)
    
    save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.__dict__,
            "golbal_step": config.global_step,
            "global_epoch": config.global_epoch,
            "train_loss": None,
            "eval_loss": None,
        }
    
    #保存当前step对应的模型文件
    #保存模型
    save_dict_and_config(save_dict = save_dict, config_dict = config.to_dict(), checkpoint_dir = checkpoint_dir, checkpoint_file = checkpoint_file)
# %% 定义训练函数
import time
def train(model: nn.Module, config: TrainingConfig, train_dataloader: DataLoader, criterion: nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, pbar: tqdm) -> float:
    model.train()
    total_loss = 0
    loss_accumulator = 0
    global resume_from_checkpoint_flag
    for cur_step, batch in enumerate(train_dataloader):
        step = cur_step + 1
        if resume_from_checkpoint_flag:
            if step < (config.global_step % len(train_dataloader)):
                continue
            else:
                resume_from_checkpoint_flag = False

        batch = data_collator(batch)
        batch = {k: v.to(config.device) for k, v in batch.items()}

        # 前向传播
        optimizer.zero_grad()
        output = model(batch["input_ids"])
        loss = criterion(output, batch["labels"])

        if step % config.gradient_accumulation_steps == 0:
            loss_accumulator += loss/config.gradient_accumulation_steps
            # 反向传播
            loss_accumulator.backward()
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 重置损失累加器
            total_loss += loss_accumulator
            loss_accumulator = 0
        else:
            loss_accumulator += loss/config.gradient_accumulation_steps
        
        config.add_global_step()
        pbar.update()
        if config.global_step % config.step_log == 0:
            # print(f"Step {step}: Loss = {total_loss / step}")
            print(f"Step {step}: Global Step{config.global_step}: Loss = {loss}")
        
        if config.global_step % config.save_step == 0:
            save_model_intrain(model, config, optimizer, scheduler)

    return total_loss / len(train_dataloader)


# %% 定义评估函数
from sklearn.metrics import accuracy_score, f1_score, recall_score 
def evaluate(model: nn.Module, config: TrainingConfig, eval_dataloader: DataLoader, criterion: nn.modules.loss) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        labels = []
        predictions = []
        for cur_step, batch in enumerate(eval_dataloader):
            step = cur_step + 1
            batch = data_collator(batch)
            batch = {k: v.to(config.device) for k, v in batch.items()}

            output = model(batch["input_ids"])
            loss = criterion(output, batch["labels"])
            total_loss += loss.item()

            labels = labels + batch["labels"].tolist()
            predictions = predictions + output.argmax(dim=1).tolist()

        accuracy = accuracy_score(labels, predictions)  
    
        # 计算F1分数（假设是二分类或多分类问题，可能需要指定average参数）  
        f1 = f1_score(labels, predictions, average='macro')  # 或者 'macro', 'micro' 等  
            
        # 计算召回率  
        recall = recall_score(labels, predictions, average='macro')  # 或者 'macro', 'micro' 等

        if accuracy > config.best_eval_indicator:
            print("New best model found!!!!")
            best_save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.__dict__,
                "golbal_step": config.global_step,
                "global_epoch": config.global_epoch,
                "train_loss": None,
                "eval_loss": total_loss / len(eval_dataloader),
            }
            best_config = config.to_dict()
            config.set_best_eval_indicator(accuracy)
            model_save_dir = config.model_save_dir
            best_dir = os.path.join(model_save_dir, f'best_model/')  
            best_file = f'best_model.pth'
            save_dict_and_config(save_dict = best_save_dict, config_dict = best_config, checkpoint_dir = best_dir, checkpoint_file = best_file)

        
        eval_indicator = {
            'loss':total_loss / len(eval_dataloader),
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            }


    return eval_indicator

# %% 定义保存模型和配置的函数

def save_model_inepoch(model: nn.Module, config: TrainingConfig, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, train_loss: float, eval_loss: float) -> None:
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.__dict__,
        "train_loss": train_loss,
        "eval_loss": eval_loss
    }

    save_dict_and_config(save_dict = save_dict, config_dict = config.to_dict(), checkpoint_dir = config.model_save_dir, checkpoint_file = "model.pth")

        


# %% 训练模型

model.to(config.device)
for epoch in range(start_epoch, config.epochs):
    print(f"Epoch {epoch}:")
    train_loss = train(model, config, train_dataloader,
                       criterion, optimizer, scheduler, pbar)
    print(f"Train Loss: {train_loss}")
    eval_indicator = evaluate(model, config, eval_dataloader, criterion)
    print(f"Eval Loss: {eval_indicator}")

    if config.global_epoch % config.save_epochs == 0:
        save_model_inepoch(model, config, optimizer, scheduler, train_loss, eval_indicator['loss'])
    config.add_global_epoch()
    config.set_random_seed(config.random_seed + 1)
save_model_inepoch(model, config, optimizer, scheduler, train_loss, eval_indicator['loss'])

# %% 加载模型和配置

# 读取本地模型和配置
save_dict = torch.load(os.path.join(config.model_save_dir, "model.pth"))
config_new = TrainingConfig(**save_dict["config"])
print(config_new)
model.load_state_dict(save_dict["model"])
optimizer.load_state_dict(save_dict["optimizer"])
scheduler.load_state_dict(save_dict["scheduler"])

for batch in train_dataloader:
    print(batch)
    batch = data_collator(batch)
    batch = {k: v.to(config.device) for k, v in batch.items()}
    print(batch)
    output = model(batch["input_ids"])
    print(output)
    idx_output = output.argmax(dim=1)
    print(batch["input_ids"], idx_output)
    break
# %%定义Trainer及其参数
# 但是不能直接这样去使用huggingface的这套工作流，因为它只适合它自己的模型，PretrainModel那个类
# # 定义训练参数
# args = TrainingArguments(
#     output_dir=r'D://code//python//outputtest//',
#     per_device_train_batch_size=5,
#     per_device_eval_batch_size=5,
#     gradient_accumulation_steps=2,
#     num_train_epochs=8,
#     weight_decay=0.1,
#     ddp_find_unused_parameters=False,
#     warmup_steps=0,
#     learning_rate=1e-4,
#     evaluation_strategy="steps",
#     eval_steps=50,
#     save_steps=50,
#     save_strategy="steps",
#     save_total_limit=1,
#     report_to="tensorboard",
#     optim="adamw_torch",
#     lr_scheduler_type="cosine",
#     bf16=True,

#     logging_steps=10,
#     log_level="info",
#     # logging_first_step=True,
#     load_best_model_at_end=True,   # 训练结束时加载最佳模型
#     metric_for_best_model='eval_loss',  # 最佳模型的评估指标
#     greater_is_better=False,       # 评估指标越低越好
#     # group_by_length=True,
#     # deepspeed='./ds_config_one_gpu.json',
# )

# # 定义Trainer


# class TwoLayerTrainer(Trainer):
#     def __init__(self, **kwargs):
#         super(TwoLayerTrainer, self).__init__(**kwargs)
#         # 创建交叉熵损失函数
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def compute_loss(self, model, inputs, return_outputs=False):
#         # 获取模型的输出
#         # {"input_ids": input_ids, "labels": labels}
#         outputs = model(input["input_ids"])

#         loss = self.criterion(outputs, input["labels"])
#         return (loss, outputs) if return_outputs else loss


# trainer = TwoLayerTrainer(model=model, args=args, train_dataset=train_dataset,
#                           eval_dataset=eval_dataset, data_collator=data_collator)

# # %% 训练模型
# trainer.train()
