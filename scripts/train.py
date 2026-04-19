import os
import yaml
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

def format_prompts(examples):
    """
    Transforms the raw text and labels into an instruction format suitable for causal language models.
    """
    instructions = []
    for text, label in zip(examples['text'], examples['label']):
        prompt = f"Categorise the intent of the following banking query.\n\nQuery: {text}\n\nIntent ID: {label}"
        instructions.append(prompt)
    return {"text": instructions}

def main():
    print("Loading hyperparameters from configuration...")
    config_path = "configs/train.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("Initialising Unsloth model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model_name"],
        max_seq_length = config["max_seq_length"],
        dtype = None, 
        load_in_4bit = True, 
    )

    print("Applying LoRA (Parameter-Efficient Fine-Tuning)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora_arguments"]["r"],
        target_modules = config["lora_arguments"]["target_modules"],
        lora_alpha = config["lora_arguments"]["lora_alpha"],
        lora_dropout = config["lora_arguments"]["lora_dropout"],
        bias = config["lora_arguments"]["bias"],
        use_gradient_checkpointing = config["lora_arguments"]["use_gradient_checkpointing"],
    )

    print("Loading preprocessed training data...")
    df_train = pd.read_csv("sample_data/train.csv")
    train_dataset = Dataset.from_pandas(df_train)
    
    # Map the formatting function over the dataset
    train_dataset = train_dataset.map(format_prompts, batched=True)

    print("Configuring the SFTTrainer...")
    output_dir = config['training_arguments']['output_dir']
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config["max_seq_length"],
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = config["training_arguments"]["per_device_train_batch_size"],
            gradient_accumulation_steps = 4,
            warmup_steps = config["training_arguments"]["warmup_steps"],
            num_train_epochs = config["training_arguments"]["num_train_epochs"],
            learning_rate = float(config["training_arguments"]["learning_rate"]),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = config["training_arguments"]["optim"],
            weight_decay = config["training_arguments"]["weight_decay"],
            lr_scheduler_type = config["training_arguments"]["lr_scheduler_type"],
            seed = config["training_arguments"]["seed"],
            output_dir = output_dir,
        ),
    )

    print("Commencing fine-tuning...")
    trainer_stats = trainer.train()
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")

    save_path = config['save_model_path']
    print(f"Saving model checkpoint to {save_path}...")
    
    model.save_pretrained(save_path) 
    tokenizer.save_pretrained(save_path)
    print("Process finished successfully.")

if __name__ == "__main__":
    main()