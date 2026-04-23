import os
import yaml
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from tqdm import tqdm

def evaluate_model(model, tokenizer, df_test):
    print("Switching model to inference mode...")
    FastLanguageModel.for_inference(model)
    
    y_true = []
    y_pred = []
    total_samples = len(df_test)
    
    for _, row in tqdm(df_test.iterrows(), total=total_samples, desc="Evaluating"):
        text = row['text']
        true_label = int(row['label']) 
        
        prompt = f"Categorise the intent of the following banking query.\n\nQuery: {text}\n\nIntent ID: "
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        input_length = inputs.input_ids.shape[1]
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id,
            max_length=None
        )
        
        prediction_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        numeric_id = ''.join(filter(str.isdigit, prediction_text))
        
        y_true.append(true_label)
        y_pred.append(int(numeric_id) if numeric_id else -1)
            
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print(f"\nFinal Test Set Results ({total_samples} samples):")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro F1:  {macro_f1 * 100:.2f}%")
    print(f"Micro F1:  {micro_f1 * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))



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
        device_map = {"": 0}
    )
    
    def format_prompts(examples):
        """
        Transforms the raw text and labels into an instruction format suitable for causal language models.
        """
        instructions = []
        for text, label in zip(examples['text'], examples['label']):
            # Adding the EOS token helps the model learn where to stop
            prompt = f"Categorise the intent of the following banking query.\n\nQuery: {text}\n\nIntent ID: {label}{tokenizer.eos_token}"
            instructions.append(prompt)
        return {"text": instructions}

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
    df_train = pd.read_csv("sample_data/train.csv") # Modify if needed
    train_dataset = Dataset.from_pandas(df_train)
    
    print("Loading test data")
    df_test = pd.read_csv("sample_data/test.csv") # Modify if needed
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
            max_length = config["max_seq_length"],
            dataset_num_proc = 2,
            packing = False, 
            per_device_train_batch_size = config["training_arguments"]["per_device_train_batch_size"],
            gradient_accumulation_steps = config["training_arguments"]["gradient_accumulation_steps"],
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
            output_dir = config['training_arguments']['output_dir'],
            eval_strategy = "epoch",
            save_strategy = "no"
        ),
    )


    print("Commencing fine-tuning")
    trainer_stats = trainer.train()
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")

    save_path = config['save_model_path']
    print(f"Saving model checkpoint to {save_path}.")
    
    model.save_pretrained(save_path) 
    tokenizer.save_pretrained(save_path)
    print("Process finished successfully.")
    
    print("Evaluate Model")
    evaluate_model(model, tokenizer, df_test)

if __name__ == "__main__":
    main()