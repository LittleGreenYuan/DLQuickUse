import json
import os
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% 定义数据集
# from datasets import Dataset, load_dataset
from torch.utils.data import Dataset

train_inputs_ids = torch.randint(0, 10, (200,), dtype=torch.long)
train_outputs_ids = (train_inputs_ids + 3) % 10


eval_inputs_ids = torch.randint(0, 10, (100,), dtype=torch.long)
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

        # self.random_seed = torch.get_rng_state()
        self.random_seed: int = 233

        # 使用 kwargs 来覆盖默认值（如果存在的话）
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        torch.manual_seed(self.random_seed)

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

    def add_global_step(self):
        self.global_step += 1

    def add_global_epoch(self):
        self.global_epoch += 1

    def to_dict(self):
        # 转换为字典，但不包括 device，因为它会在加载时重新计算
        config_dict = {
            key: getattr(self, key)
            for key in vars(self).keys() if key != 'device'
        }
        '''     
        config_dict = {  
            key: getattr(self, key) 
            for key in [  
                'learning_rate', 'batch_size', 'train_dataloader_shuffle',  
                'eval_dataloader_shuffle', 'epochs', 'model_save_dir',  
                'step_log', 'global_step', 'global_epoch', 'save_epochs'  
            ]  
        }
        '''
        return config_dict

    def __str__(self):
        # 提供一个字符串表示，方便打印配置信息
        return f"TrainingConfig(learning_rate={self.learning_rate}, batch_size={self.batch_size}, epochs={self.epochs}, model_save_path='{self.model_save_dir}, device='({self.device})', train_dataloader_shuffle={self.train_dataloader_shuffle}, eval_dataloader_shuffle={self.eval_dataloader_shuffle}, show_step={self.step_log}), global_step={self.global_step}, global_epoch={self.global_epoch}, save_epochs={self.save_epochs}, random_seed={self.random_seed}"


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

# %% 定义训练函数

def train(model: nn.Module, config: TrainingConfig, train_dataloader: DataLoader, criterion: nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler) -> float:
    model.train()
    total_loss = 0
    for cur_step, batch in enumerate(train_dataloader):
        step = cur_step + 1
        batch = data_collator(batch)
        batch = {k: v.to(config.device) for k, v in batch.items()}

        # 前向传播
        optimizer.zero_grad()
        output = model(batch["input_ids"])
        loss = criterion(output, batch["labels"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        config.add_global_step()
        if config.global_step % config.step_log == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    return total_loss / len(train_dataloader)


# %% 定义评估函数

def evaluate(model: nn.Module, config: TrainingConfig, eval_dataloader: DataLoader, criterion: nn.modules.loss) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for cur_step, batch in enumerate(eval_dataloader):
            step = cur_step + 1
            batch = data_collator(batch)
            batch = {k: v.to(config.device) for k, v in batch.items()}

            output = model(batch["input_ids"])
            loss = criterion(output, batch["labels"])
            total_loss += loss.item()

    return total_loss / len(eval_dataloader)

# %% 定义保存模型和配置的函数

def save_model(model: nn.Module, config: TrainingConfig, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, train_loss: float, eval_loss: float) -> None:
    if config.global_epoch % config.save_epochs == 0:
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.__dict__,
            "train_loss": train_loss,
            "eval_loss": eval_loss
        }
        torch.save(save_dict, os.path.join(config.model_save_dir, "model.pth"))
        print("\nModel saved.")
        # 保存训练配置
        # config_dict = config.__dict__.copy()
        # # 删除 device 属性（假设它存在）
        # if 'device' in config_dict:
        #     del config_dict['device']
        with open(os.path.join(config.model_save_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f)
            print("\nconfig saved.")
            print("--------------------------------------------------------------------")


# %% 训练模型

model.to(config.device)
for epoch in range(config.epochs):
    print(f"Epoch {epoch+1}:")
    train_loss = train(model, config, train_dataloader,
                       criterion, optimizer, scheduler)
    print(f"Train Loss: {train_loss}")
    eval_loss = evaluate(model, config, eval_dataloader, criterion)
    print(f"Eval Loss: {eval_loss}")

    config.add_global_epoch()

    save_model(model, config, optimizer, scheduler, train_loss, eval_loss)

# %% 加载模型和配置

# 读取本地模型和配置
save_dict = torch.load(os.path.join(config.model_save_dir, "model.pth"))
config = TrainingConfig(**save_dict["config"])
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
