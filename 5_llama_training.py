#предыдущий файл - 4_train_test_splitting.ipynb
#большая часть параметров модели и треинера подбиралась вручную экспериментально на сервере, чтобы эффективно обучить модель, используя предоставленные ресурсы сервера

import os
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()

MODEL_NAME = "KingNish/Reasoning-Llama-1b-v0.1"
DATASET_PATHS = ["train_dataset_1.csv", "train_dataset_2.csv"]
OUTPUT_DIR = "./final_model"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 6
LEARNING_RATE = 3e-4
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

special_tokens = ["<|result|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
tokenizer.add_special_tokens({"additional_special_tokens": [t for t in special_tokens if t not in tokenizer.additional_special_tokens]})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj"],
    task_type="CAUSAL_LM",
    bias="none",
    fan_in_fan_out=True
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

def load_dataset(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep='\t')
        df['full_text'] = df.apply(lambda row: f"{row['text'].strip()} <|result|>", axis=1)
        df['labels'] = df.apply(lambda row: row['answer'].strip(), axis=1)
        dfs.append(df)
    return pd.concat(dfs)

train_df = load_dataset(DATASET_PATHS)

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_function(examples):
        full_texts = [f"{text.strip()} {label.strip()}" for text, label in zip(examples["full_text"], examples["labels"])]
            
        model_inputs = tokenizer(
                                full_texts,
                                        max_length=MAX_SEQ_LENGTH,
                                                truncation=True,  padding="max_length")
                    
        model_inputs["labels"] = model_inputs["input_ids"].copy()
                        
        return model_inputs

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=2500,
    save_steps=5000,
    save_total_limit=2,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_8bit",
    fp16=False,
    bf16=True,
    tf32=True,
    report_to="none",
    remove_unused_columns=False,
    resume_from_checkpoint=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "merged_model"))

#следующий файл - 6_inference.py