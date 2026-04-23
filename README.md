# Banking Intent Detection Fine-Tuning with Unsloth

This repository contains the source code and configuration files for Project 2 of the Applications of Natural Language Processing in Industry module at the University of Science, Faculty of Information Technology.

The primary objective of this project is to study and apply fine-tuning techniques to a banking intent classification task using the BANKING77 dataset and the Unsloth library.

---

## Quick Start: Run on Kaggle (Recommended)

To avoid local hardware limitations and complex environment setup, we strongly recommend running this project directly in our pre-configured Kaggle Notebook.

Kaggle provides free access to NVIDIA T4 and P100 GPUs, which are required for Unsloth's 4-bit quantised fine-tuning.

**[Run the complete pipeline on Kaggle here](https://www.kaggle.com/code/thoangha/unslothbankingintent)**

## Repository Structure

The project is structured following the required format to ensure logical separation of scripts, configurations, and data:

```text
banking-intent-unsloth/
|-- scripts/
|   |-- train.py
|   |-- inference.py
|   |-- preprocess_data.py
|-- configs/
|   |-- train.yaml
|   |-- inference.yaml
|-- sample_data/
|   |-- train.csv
|   |-- test.csv
|-- train.sh (for Linux / MacOS / WSL)
|-- inference.sh (for Linux / MacOS / WSL)
|-- train.bat (for Windows)
|-- inference.bat (for Windows)
|-- requirements.txt
|-- README.md
```

# CRITICAL WARNING: NVIDIA GPU REQUIRED

The Unsloth library is heavily optimised for CUDA hardware and will not run on a standard CPU.

If your local machine does not have a dedicated NVIDIA GPU with sufficient memory (at least 15 GB VRAM is recommended), we strongly redirect you to run this project on Kaggle. You can easily upload these scripts to a Kaggle Notebook and utilise their free P100 or T4x2 GPU accelerators to complete the training seamlessly.

## 1. Environment Setup

### Step 1. Clone the Repository

```bash
git clone https://github.com/THoangHa/UnslothBankingIntent.git
cd UnslothBankingIntent
```

### Step 2. Create and Activate Virtual Environment

For Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

For Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3. Install Dependencies

```bash
pip install -r requirements.txt
```

(Note: Unsloth has specific installation requirements depending on your CUDA version. If you encounter issues, please refer to the official [Unsloth GitHub Repository](https://github.com/unslothai/unsloth#install-unsloth).)

## 2. Downloading Data and Models

### Data preparation

If the `sample_data/` directory is empty, you must generate the formatted datasets. Ensure your raw BANKING77 data is accessible, then run the preprocessing script. This will construct the `train.csv` and `test.csv` files with the correct text and label mappings. Each file accounts for 10% of the total dataset size to save time and computation resources.

```bash
python scripts/preprocess_data.py
```

### Model downloading

You do not need to download the base Large Language Model manually. The training script uses the `FastLanguageModel.from_pretrained()` method, which will automatically download the 4-bit quantised base model (e.g., `unsloth/llama-3-8b-bnb-4bit`) from the Hugging Face Hub during the first execution.

## 3. Training the Model

The training pipeline is controlled by the parameters specified in `configs/train.yaml`. You can modify this file to adjust the batch size, learning rate, number of epochs, and LoRA target modules.
To commence fine-tuning, make the shell script executable and run it:

For Windows

```bash
python train.bat
```

For Linux / macOS / WSL

```bash
chmod +x train.sh
./train.sh
```

What this script does:

1. Loads the base LLM in 4-bit precision.
2. Applies the LoRA adapters for parameter-efficient training.
3. Formats the datasets and sets up the `SFTTrainer`.
4. Trains the model across the designated number of epochs.
5. Evaluates the final trained model against the test dataset, calculating exact classification metrics (Accuracy, F1-scores, and a classification report).
6. Saves the final fine-tuned adapter weights and tokenizer to the directory specified in `configs/train.yaml` (typically `./saved_model`).

## 4. Inference

Once the model has been successfully fine-tuned and saved, you can evaluate its performance on the unseen test set. The inference parameters are controlled by `configs/inference.yaml`.

To execute the evaluation pipeline, make the shell script executable and run it:

For Windows

```bash
python inference.bat
```

For Linux / macOS / WSL

```bash
chmod +x inference.sh
./inference.sh
```

What this script does:

1. Initialises the IntentClassification pipeline.
2. Loads the base model and merges it with your fine-tuned LoRA adapters from the `./saved_model` directory.
3. Switches the model to inference mode.
4. Feeds a sample test message to the model and prints the predicted banking intent directly to the console.
