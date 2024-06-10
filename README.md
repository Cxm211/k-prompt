# K-prompt: Adapting Kownledge Prompt Tuning for Enhanced Automated Program Repair

## Overview
This repo contains all code, data and results for the paper "K-prompt: Adapting Kownledge Prompt Tuning for Enhanced Automated Program Repair" 

This README provides a comprehensive guide for users to understand and execute the script efficiently. Adjust any specifics to your project's requirements or preferences.

<!-- ## Features
- Training, evaluation, and testing functionalities.
- Customizable training epochs, batch sizes, learning rates, and more through command-line arguments.
- Support for CUDA and multi-GPU training.
- Detailed logging of training and evaluation metrics. -->

## Requirements
- Python 3.x
- PyTorch
- Transformers
- OpenPrompt
- tqdm
- numpy

## Setup
Ensure that all required libraries are installed. You can install them using the command:
   ```bash
    pip install -r requirements.txt
```

## Usage
This project includes several Python scripts, hard_codet5p.py is for hard prompt while soft_codet5p.py is for soft prompt, and finetune_t5_gene is for fine-tuning.
To run the script, 
1. If the script is hard_codet5p.py or soft_codet5p.py, make sure correct script path and lib file are used. Change the lib file according to the dataset here:
```python
from my_lib_tfix import read_prompt_examples, ...
```
2. use the following command in terminal for example:
```bash
    python hard_codet5p.py --log_name <log_name> --model_name <log_name> --do_train --do_eval  --do_test  --lang <programming_language> --data_dir <dataset_directory> --output_dir <output_path>

```

## Project Structure

This project is organized in the following way:

### `/data`
- **Description**: Contains all the training, validation, and testing data. 
- **Subfolders**:
  - `/tfix`: Dataset for TFix dataset.

### `/scripts`
- **Description**: Stores scripts of hard and soft prompts used in paper
- **Examples**:
  - `hard_template_tfix.txt`: Hard prompt template file for the TFix dataset.

### `/results`
- **Description**: Stores all results of our experiments

