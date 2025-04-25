################################################################
#   BLEU Scores:
#   BLEU-1: 0.4639
#   BLEU-2: 0.1467
#   BLEU-3: 0.0999
#   BLEU-4: 0.0825
######################################################################

from transformers import T5Tokenizer,T5ForConditionalGeneration
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import json
from  torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForCausalLM
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



# 评估BLUE-1，BLUE-2，BLUE-3，BLUE-4
def evaluate_blue(model, test_data_loader):
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    smoothing = SmoothingFunction().method1
    bar=tqdm(test_data_loader)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(bar):
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128
            )
            preds=tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
            references=tokenizer.batch_decode(labels, skip_special_tokens=True)
            print("\n======================================\n")
            print(preds)
            print("\n")
            print(references)
            print("\n======================================\n")
            for pred, ref in zip(preds, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                for n in range(1, 5):
                    bleu = sentence_bleu(
                        ref_tokens,
                        pred_tokens,
                        weights=tuple([1.0/n]*n),
                        smoothing_function=smoothing
                    )
                    bleu_scores[n].append(bleu)
        avg_bleu = {n: sum(scores)/len(scores) for n, scores in bleu_scores.items()}
    return avg_bleu
if __name__=='__main__':
    batch_size=4
    model=T5ForConditionalGeneration.from_pretrained('model')
    model.load_state_dict(torch.load("/workspace/projects/Bob_llama/Finetuning_T5/output/model.pth"))
    tokenizer=T5Tokenizer.from_pretrained('model')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    test_data=load_data("data/dev.json")
    test_data_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True,collate_fn=batchdata_fuc)
    blue_scores=evaluate_blue(model,test_data_loader)
    # Print results
    print("\nBLEU Scores:")
    for n, score in blue_scores.items():
        print(f"BLEU-{n}: {score:.4f}")