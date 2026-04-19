import os
import yaml
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from tqdm import tqdm

def format_prompts(examples):
    """
    Transforms the raw text and labels into an instruction format suitable for causal language models.
    """
    instructions = []
    for text, label in zip(examples['text'], examples['label']):
        prompt = f"Categorise the intent of the following banking query.\n\nQuery: {text}\n\nIntent ID: {label}"
        instructions.append(prompt)
    return {"text": instructions}

def calculate_accuracy(model, tokenizer, df_test):
    """
    Runs an inference loop over the test set to calculate exact classification accuracy.
    """
    print("\nCalculating final accuracy on the test set")
    
    FastLanguageModel.for_inference(model)
    
    correct_predictions = 0
    total_samples = len(df_test)
    
    for _, row in tqdm(df_test.iterrows(), total=total_samples, desc="Evaluating"):
        text = row['text']
        true_label = str(row['label'])
        
        # Format identical to training
        prompt = f"Categorise the intent of the following banking query.\n\nQuery: {text}\n\nIntent ID:"
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        try:
            # Extract the predicted number
            predicted_id_str = decoded_output.split("Intent ID:")[-1].strip()
            predicted_id = ''.join(filter(str.isdigit, predicted_id_str))
        except (ValueError, IndexError):
            predicted_id = "-1"
            
        if predicted_id == true_label:
            correct_predictions += 1
            
    accuracy = (correct_predictions / total_samples) * 100
    
    print(f"Final Test Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
 

def main():
    print("Loading hyperparameters from configuration")
    config_path = "configs/train.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("Initialising Unsloth model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model_name"],
        max_seq_length = config["max_seq_length"],
        dtype = None, 
        load_in_4bit = True, 
    )

    print("Applying LoRA (Parameter-Efficient Fine-Tuning)")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora_arguments"]["r"],
        target_modules = config["lora_arguments"]["target_modules"],
        lora_alpha = config["lora_arguments"]["lora_alpha"],
        lora_dropout = config["lora_arguments"]["lora_dropout"],
        bias = config["lora_arguments"]["bias"],
        use_gradient_checkpointing = config["lora_arguments"]["use_gradient_checkpointing"],
    )

    print("Loading preprocessed training data")
    df_train = pd.read_csv("sample_data/train.csv")
    train_dataset = Dataset.from_pandas(df_train)
    
    print("Loading test data")
    df_test = pd.read_csv("sample_data/test.csv")
    test_dataset = Dataset.from_pandas(df_test)
    
    # Map the formatting function over the dataset
    train_dataset = train_dataset.map(format_prompts, batched=True)
    test_dataset = test_dataset.map(format_prompts, batched=True)

    print("Configuring the SFTTrainer")
    output_dir = config['training_arguments']['output_dir']
    
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = config["max_seq_length"],
            dataset_num_proc = 2,
            packing = False, 
            per_device_train_batch_size = config["training_arguments"]["per_device_train_batch_size"],
            gradient_accumulation_steps = 4,
            warmup_steps = config["training_arguments"]["warmup_steps"],
            num_train_epochs = config["training_arguments"]["num_train_epochs"],
            learning_rate = float(config["training_arguments"]["learning_rate"]),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            eval_strategy = "epoch",
            optim = config["training_arguments"]["optim"],
            weight_decay = config["training_arguments"]["weight_decay"],
            lr_scheduler_type = config["training_arguments"]["lr_scheduler_type"],
            seed = config["training_arguments"]["seed"],
            output_dir = output_dir,
            average_tokens_across_devices = False,
        ),
    )


    print("Commencing fine-tuning")
    trainer_stats = trainer.train()
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")
    
    calculate_accuracy(model, tokenizer, df_test)

    save_path = config['save_model_path']
    print(f"Saving model checkpoint to {save_path}.")
    
    model.save_pretrained(save_path) 
    tokenizer.save_pretrained(save_path)
    print("Process finished successfully.")

if __name__ == "__main__":
    main()