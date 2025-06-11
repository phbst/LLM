import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import transformers
import json
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import AdamW,get_scheduler,get_linear_schedule_with_warmup


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path="/workspace/projects/Bob_llama/Finetuning_T5/model"
train_data_path="/workspace/projects/Bob_llama/Finetuning_T5/data/train.json"
rank=16
alpha=32
dropout=0.01
learning_rate=0.001
targets=["q","k","v"]
batch_size=1
num_epochs=1
save_step=500
num_epochs=1
learning_rate=1e-5

valid_step=1000


class LoraLinear(nn.Module):
    def __init__(self, baselinear, rank, alpha, dropout):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.base_linear = copy.deepcopy(baselinear)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.base_linear.in_features, dtype=self.base_linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(self.base_linear.out_features, self.rank, dtype=self.base_linear.weight.dtype))
        nn.init.normal_(self.lora_A, mean=0.02)
        nn.init.zeros_(self.lora_B)
        for param in self.base_linear.parameters():
            param.requires_grad = False
        self.weight = self.base_linear.weight
        self.bias = self.base_linear.bias

    def forward(self, x):
        scaling = self.alpha / self.rank
        m = F.linear(self.dropout(x), self.lora_A)
        m = F.linear(m, self.lora_B)
        return self.base_linear(x) + scaling * m

def get_lora_model(module, rank, alpha, dropout):
    for name, child in module.named_children():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            for param in child.parameters():
                param.requires_grad = False
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, rank, alpha, dropout)
            setattr(module, name, lora_linear)
        else:
            get_lora_model(child, rank, alpha, dropout)
    
    return module  # 返回修改后的模块


# class LoraLinear(nn.Module):
#     def __init__(self, baselinear, rank, alpha, dropout):
#         super().__init__()
#         self.rank = rank
#         self.alpha = alpha
#         self.dropout = nn.Dropout(dropout)
#         self.base_linear = copy.deepcopy(baselinear)
        
#         # 使用 nn.Linear 替代直接参数
#         # A 矩阵: rank x in_features
#         self.lora_A = nn.Linear(self.base_linear.in_features, self.rank, bias=False)
#         # B 矩阵: rank x out_features
#         self.lora_B = nn.Linear(self.rank, self.base_linear.out_features, bias=False)
        
#         # 初始化
#         nn.init.normal_(self.lora_A.weight, mean=0, std=0.02)
#         nn.init.zeros_(self.lora_B.weight)
        
#         # 冻结基础模型参数
#         for param in self.base_linear.parameters():
#             param.requires_grad = False
            
#     def forward(self, x):
#         scaling = self.alpha / self.rank
        
#         # 使用 lora_A 和 lora_B 的前向传播
#         m = self.lora_B(self.dropout(self.lora_A(x)))
        
#         return self.base_linear(x) + scaling * m
  
  
# def get_lora_model(module, rank, alpha, dropout):
#     for name, child in module.named_children():
#         if any(s in name for s in ["embed", "norm", "lm_head"]):
#             for param in child.parameters():
#                 param.requires_grad = False
#         elif isinstance(child, nn.Linear):
#             lora_linear = LoraLinear(child, rank, alpha, dropout)
#             setattr(module, name, lora_linear)
#         else:
#             get_lora_model(child, rank, alpha, dropout)
    
#     return module  # 返回修改后的模块


def process_data(example):
    batch_inputs,batch_labels = [],[]
    for i in  example:
        batch_inputs.append(i['inputs'])
        batch_labels.append(i['output'])
    batch_inputs=tokenizer(batch_inputs,padding=True,truncation=True,return_tensors="pt")
    batch_labels=tokenizer(batch_labels,padding=True,truncation=True,return_tensors="pt")
    batch_data={
        "input_ids": batch_inputs["input_ids"],
        "attention_mask":batch_inputs["attention_mask"],
        "labels":batch_labels["input_ids"]
    }
    return batch_data


def load_data(path):
    dataset={}
    with open(path,'r',encoding='utf-8') as f:
        
        for i,l in enumerate(f):
            l=l.strip()
            data=json.loads(l)
            inputs="Question:\n"+data['question']+"\nContext:\n"+data["context"]+"\nAnswer:\n"
            output=data["answer"]
            dataset[i]={"inputs":inputs,"output":output}
        return dataset

def train(model,train_data_loader,optimizer,scheduler,num_epochs,save_step):
    total_loss=0.0
    train_loss=[]
    model.train()
    model.to(device)
    step=1
    for i in range(num_epochs):
        process_bar=tqdm(train_data_loader,leave=False)
        for batch in process_bar:
            inputs_id=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=batch["labels"].to(device)
            optimizer.zero_grad()

            output=model(input_ids=inputs_id,attention_mask=attention_mask,labels=labels)
            loss=output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss+=loss.item()
            
            if step%save_step==0:
                save_dir = f"output/cheakpoint-{step}"
                os.makedirs(save_dir, exist_ok=True) 
                torch.save(model.state_dict(), f"{save_dir}/model.pth")
            step+=1
        
        with open("output/train_loss.json","w") as f:
            json.dump(train_loss,f)
        avg_loss = total_loss / len(train_data_loader)
        print(f"Epoch [{i+1}/{num_epochs}], Loss: {avg_loss:.4f}")




if __name__ == "__main__":
    model=T5ForConditionalGeneration.from_pretrained(model_path)
    model=get_lora_model(model,rank,alpha,dropout)
    tokenizer=T5Tokenizer.from_pretrained(model_path)
    train_data=load_data(train_data_path)
    train_data_loader=DataLoader(train_data,batch_size=1,shuffle=True,collate_fn=process_data)
    optimizer=AdamW(model.parameters(),lr=)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100,num_training_steps=len(train_data_loader)*num_epochs)
    train(model,train_data_loader,optimizer,scheduler,num_epochs,save_step)