# =======================
#   STAGE 1 TRAINING
# =======================

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import os, shutil
from google.colab import files

os.environ["WANDB_DISABLED"] = "true"

round = 3 # Replace with the current stage
# ======= CONFIG =======
comment_file = f"stage_{round}.txt"
model_name = f"./stage{round-1}" # Replace with 'Qwen/Qwen2.5-0.5B-Instruct' for stage 1
save_dir_stage1 = f"./stage_{round}-comments"
max_length = 128
# ======================


# ======= LOAD & CLEAN TEXT =======
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    samples = [s.strip() for s in text.split("\n") if s.strip()]
    return samples


print("Loading datasets...")
comment_samples = load_text(comment_file)
comment_dataset = Dataset.from_dict({"text": comment_samples})


# ======= TOKENIZER & MODEL =======
print("Loading tokenizer and model...")
# Load tokenizer from *same model folder* or fallback to base model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except:
    print("Tokenizer not found in stage folder → using base tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU - slower")


# ======= TOKENIZE =======
def tokenize_fn(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing datasets...")
tokenized_comments = comment_dataset.map(tokenize_fn, batched=True)


# ======= TRAINING ARGS =======
training_args_1 = TrainingArguments(
    output_dir=save_dir_stage1,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50,
    save_total_limit=1,
    warmup_steps=20,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

trainer_1 = Trainer(
    model=model,
    args=training_args_1,
    train_dataset=tokenized_comments,
)

print("\nStage 1: Training on single comments...")
trainer_1.train()

# ======= SAVE MODEL =======
trainer_1.save_model(save_dir_stage1)
tokenizer.save_pretrained(save_dir_stage1)
print(f"Stage 1 complete. Saved at {save_dir_stage1}")


# =======================
#   STAGE 2 TRAINING
# =======================

os.environ["WANDB_DISABLED"] = "true"

# ======= CONFIG =======
qa_file = f"stage_{round}_c.txt"
prevstage_model_dir = f"./stage_{round}-comments"   # must exist
save_dir_stage2 = f"./stage_{round}-qa"
max_length = 128
# ======================


# ======= LOAD & CLEAN TEXT =======
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if "{---}" in text:
        samples = [s.strip() for s in text.split("{---}") if s.strip()]
    else:
        samples = [line.strip() for line in text.split("\n") if line.strip()]

    return samples

print("Loading QA dataset...")
qa_samples = load_text(qa_file)
qa_dataset = Dataset.from_dict({"text": qa_samples})


# ======= LOAD MODEL FROM STAGE 1 =======
print("Loading Stage 1 model...")

# Load tokenizer from *same model folder* or fallback to base model
try:
    tokenizer = AutoTokenizer.from_pretrained(prevstage_model_dir)
except:
    print("Tokenizer not found in stage folder → using base tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    prevstage_model_dir,
    device_map="auto",
    trust_remote_code=True
)

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU - slower")


# ======= TOKENIZE =======
def tokenize_fn(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing QA dataset...")
tokenized_qa = qa_dataset.map(tokenize_fn, batched=True)


# ======= TRAINING =======
training_args_2 = TrainingArguments(
    output_dir=save_dir_stage2,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=50,
    save_total_limit=1,
    warmup_steps=20,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

trainer_2 = Trainer(
    model=model,
    args=training_args_2,
    train_dataset=tokenized_qa,
)

print("\nStage 2: Fine-tuning on QA conversations...")
trainer_2.train()


# ======= SAVE FINAL MODEL =======
trainer_2.save_model(save_dir_stage2)
tokenizer.save_pretrained(save_dir_stage2)

print(f"Final model saved at {save_dir_stage2}")

# ======= ZIP & DOWNLOAD =======
zip_path = f"stage{round}_final_model"
shutil.make_archive(zip_path, 'zip', save_dir_stage2)
