# FixIt — QLoRA Fine-tuning Notebook

import json

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item.pop("_meta", None)
                data.append(item)
    return data

train_data = load_jsonl("train.jsonl")
val_data = load_jsonl("val.jsonl")

print(f"Train: {len(train_data)} examples")
print(f"Val: {len(val_data)} examples")

print("\nExample:")
for msg in train_data[0]["messages"]:
    print(f"  [{msg['role']}]: {msg['content'][:100]}...")

# ============================================================
# Convert to Hugging Face Dataset
# ============================================================
from datasets import Dataset

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(train_dataset)
print(val_dataset)

# ============================================================
# Load the base model in 4-bit (the "Q" in QLoRA)
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit precision
    bnb_4bit_quant_type="nf4",            # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype=torch.float16, # Compute in fp16 for speed
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants too
)

print("[*] Loading model (this takes 2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",  
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Mistral doesn't have a pad token
tokenizer.padding_side = "right"

print(f"[✓] Model loaded: {MODEL_NAME}")
print(f"    GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ============================================================
# Attach LoRA adapters (the "LoRA" in QLoRA)
# ============================================================
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for training with quantization
model = prepare_model_for_kbit_training(model)

# LoRA config — these small adapter matrices are ALL we train
lora_config = LoraConfig(
    r=16,                # Rank — higher = more capacity, more VRAM
    lora_alpha=32,       # Scaling factor — typically 2*r
    lora_dropout=0.05,   # Small dropout for regularization
    bias="none",
    task_type="CAUSAL_LM",
    # Which layers to add LoRA to — attention layers are standard
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ============================================================
# Training configuration
# ============================================================
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./fixit-lora",
    
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch size = 4*4 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,              # Warm up LR for first 3% of steps
    lr_scheduler_type="cosine",     # Cosine decay after warmup
    
    # Memory optimization
    fp16=True,                       # Mixed precision training
    max_seq_length=2048,             # Context window
    gradient_checkpointing=True,     # Trade compute for memory
    
    # Logging
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,             # Keep only last 2 checkpoints
    
    # Dataset
    dataset_text_field=None,         # We use the chat template below
)

# ============================================================
# Train!
# ============================================================

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("[*] Starting training...")
print(f"    Train examples: {len(train_dataset)}")
print(f"    Val examples: {len(val_dataset)}")
print(f"    Epochs: 3")
print(f"    Effective batch size: 16")
print(f"    Estimated time: 30-45 min on T4")

trainer.train()

print("[✓] Training complete!")

# ============================================================
# Save the LoRA adapter
# ============================================================

trainer.save_model("./fixit-lora-final")
tokenizer.save_pretrained("./fixit-lora-final")

print("[✓] Adapter saved to ./fixit-lora-final")
print("    Download this folder — you'll need it for the API")

# ============================================================
# Quick test before downloading
# ============================================================
from peft import PeftModel


test_error = """RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 15.78 GiB total capacity; 13.12 GiB already allocated;
1.41 GiB free; 14.26 GiB reserved in total by PyTorch)"""

messages = [
    {"role": "system", "content": "You are FixIt, an expert ML debugging assistant. Given a PyTorch, YOLO, or CUDA error message or traceback, diagnose the root cause and provide a clear, actionable fix. Include corrected code when applicable."},
    {"role": "user", "content": f"Diagnose this error and tell me exactly how to fix it. Be specific — include the root cause and corrected code if applicable.\n\n```\n{test_error}\n```"},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=" * 60)
print("TEST RESPONSE:")
print("=" * 60)
print(response)

