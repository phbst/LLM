import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import TaskType, LoraConfig, get_peft_model

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths and hyperparameters
model_path = "/workspace/projects/Bob_llama/Finetuning_T5/model"
train_data_path = "/workspace/projects/Bob_llama/Finetuning_T5/data/train.json"
rank = 16
alpha = 32
dropout = 0.01
learning_rate = 1e-5
batch_size = 1
num_epochs = 1
save_step = 500

# Custom Dataset class
class QADataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                inputs = f"Question:\n{data['question']}\nContext:\n{data['context']}\nAnswer:\n"
                output = data["answer"]
                self.data.append({"inputs": inputs, "output": output})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["inputs"], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        labels = self.tokenizer(item["output"], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

if __name__ == "__main__":
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Adapt the model using PEFT (LoRA)
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # Changed from CAUSAL_LM to SEQ_2_SEQ_LM for T5
        target_modules=['q', 'v'],
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout
    )
    model = get_peft_model(model, config)
    model.to(device)
    
    # Load data
    train_dataset = QADataset(train_data_path, tokenizer)
    
    # Training arguments
    args = TrainingArguments(
        output_dir="./outmodel",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        save_steps=save_step,
        logging_steps=10,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True if device == 'cuda' else False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()