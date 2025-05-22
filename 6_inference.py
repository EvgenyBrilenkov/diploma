#предыдущий файл - 5_llama_training.py
#теперь необходимо провести инференс модели для подсчета метрик

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_DIR = "./final_model/merged_model"

tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

test_df = pd.read_csv("valid_dataset.csv", sep="\t")

def generate_prediction(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.split("<|result|>")[-1].strip()

predictions = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
    prompt = row['text']
    prediction = generate_prediction(prompt)
    predictions.append(prediction)

test_df['prediction'] = predictions
test_df.to_csv("inference_results.csv", index=False, sep='\t')

#следующий файл - 7_metrics.ipynb