import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset
import transformers
import json
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import AdamW,get_scheduler,get_linear_schedule_with_warmup


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def load_data(file_path):
    dataset = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:  
                line_dict = json.loads(line) 
                inputs = "question:\n" + line_dict["question"] + "\ncontext:\n" + line_dict["context"]
                label = line_dict["answer"]
                dataset[idx] = {"inputs": inputs, "label": label}
    return dataset    

def batchdata_fuc(batch_data):
    batch_inputs,batch_labels = [],[]
    for i in  batch_data:
        batch_inputs.append(i['inputs'])
        batch_labels.append(i['label'])
    batch_inputs_id=tokenizer(
        batch_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    batch_labels_id=tokenizer(
        batch_labels,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    batch_data={
        "input_ids":batch_inputs_id["input_ids"],
        "attention_mask":batch_inputs_id["attention_mask"],
        "labels":batch_labels_id
    }
    return batch_data

def plot_loss(train_path):
    
    with open(train_path,"r") as f:
        data=json.load(f)
        x=[i["step"] for i in data]
        y=[i["loss"] for i in data]
    plt.plot(x,y)
    plt.show()

def valid_model(model,test_loader):

    model.eval()
    valid_loss=[]
    total_loss=0.0
    step=1
    process_bar=tqdm(test_loader,desc="valid",leave=False)
    with torch.no_grad():
        for batch_data in  process_bar:
            inputs_id=batch_data["input_ids"].to(device)
            attention_mask=batch_data["attention_mask"].to(device)
            labels=batch_data["labels"].to(device)
            output=model(
                input_ids=inputs_id, 
                attention_mask=attention_mask, 
                labels=labels 
            )
            loss=output.loss
            total_loss+=loss.item()
            valid_loss.append({"step":step,"loss":loss.item()})
            step+=1
        mean_loss=total_loss/len(test_loader)
    model.train()
    return mean_loss


def train_model(model,train_loader,test_loader,optimizer,num_epochs,scheduler,save_step,valid_step):
    total_loss = 0.0
    train_loss=[]
    model.train()
    model.to(device)
    step=1
    for epoch in range(num_epochs):
        process_bar=tqdm(train_loader,desc=f"train-epoch{epoch}",leave=False)
        for batch in process_bar:
            inputs_id=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=batch["labels"].to(device)

            optimizer.zero_grad()

            outputs=model(input_ids=inputs_id,attention_mask=attention_mask,labels=labels)

            loss=outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss+=loss.item()
            print(f"===========Step:{step}-----Loss: {loss.item():.4f}===================")
            train_loss.append({"step":step,"loss":loss.item()})

            if step%valid_step==0:
                valid_loss=valid_model(model,test_loader)
                print(f"\n========================\n\nvalid_loss:{valid_loss}\n\n===========================\n")

            if step%save_step==0:
                save_dir = f"output/cheakpoint-{step}"
                os.makedirs(save_dir, exist_ok=True) 
                torch.save(model.state_dict(), f"{save_dir}/model.pth")
                
            step+=1

        with open("output/train_loss.json","w") as f:
            json.dump(train_loss,f)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    num_epochs=1
    learning_rate=1e-5
    batch_size=2
    save_step=2000
    valid_step=1000
    
    tokenizer=  T5Tokenizer.from_pretrained("model")
    model= T5ForConditionalGeneration.from_pretrained('model')

    print("\n================model init success!======================")
    print(f"\nparamerts:{sum(p.numel() for p in model.parameters())}")

    test_data=load_data("data/dev.json")
    train_data=load_data("data/train.json")

    train_data_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=batchdata_fuc)
    test_data_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True,collate_fn=batchdata_fuc)

    print("\n==========================data init success=================")
    print(f"\ndata_example:{train_data[0]}")

    optimizer=AdamW(model.parameters(),lr=learning_rate) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_data_loader) * num_epochs
    )
    print("\n======================train config=====================")
    print(f"\n学习率:{learning_rate}\nbatch_size:{batch_size}\nnum_epochs:{num_epochs}\ntrain_step:{num_epochs*len(train_data)/batch_size}\nsave_step:{save_step}\nvalid_step:{valid_step}")
    print("\n======================train start=====================")
    train_model(model,train_data_loader,test_data_loader,optimizer,num_epochs,scheduler,save_step,valid_step)
    
    torch.save(model.state_dict(),"output/model.pth")
    tokenizer.save_pretrained("model")

    plot_loss("output/train_loss.json")
    