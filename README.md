# K-prompt: Adapting Kownledge Prompt Tuning for Enhanced Automated Program Repair

## Overview
This repo contains all code, data and results for the paper "Adapting Kownledge Prompt Tuning for Enhanced Automated Program Repair" 

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
Then install the OpenPrompt tool:
   ```bash
    cd OpenPrompt
    pip install .
```
## Evaluation Metrics

### Exact Match (EM):

EM refers to the generated patches that exactly match the fix reference. 
Since the correctness of code in the majority of programming languages, except for Python, is not affected by the additional whitespaces, we remove extra whitespaces when measuring EM for programming languages that are not affected.

### Syntactically Correct Patch (SC):
Since some buggy programs may have more than one correct fix, in addition to EM, we define the generated patches that are syntactically equivalent to the fixed reference as syntactically correct patches by comparing their syntax tree. 

### CodeBLEU:
In addition to these two metrics, EM and SC, a more relaxed metric, \textit{CodeBLEU}, is included to measure the extent to which a program is repaired. Unlike the traditional BLEU score, which is a metric commonly used in NLP tasks to measure the closeness of a model-generated response to the ground truth text, 
CodeBLEU more accurately measures the similarity between generated patches and reference fixes by taking both syntax and semantics of programs into account. 

A CodeBLEU score is often calculated as:
$CodeBLEU = \alpha \cdot BLEU + \beta \cdot BLEU_{weight} + \gamma \cdot Match_{ast} + \delta \cdot Match_{df}$,
where ${\displaystyle BLEU}$ is calculated by standard n-gram BLEU and ${\displaystyle BLEU_{weight}}$ is the weighted n-gram match, obtained by comparing the generated patch tokens and the fix reference tokens with different weights. ${\displaystyle Match_{ast}}$ and ${\displaystyle Match_{df}}$ the syntactic AST match and the semantic dataflow match, representing the syntactic information of code and the semantic similarity between the generated patch tokens and the fix reference tokens, respectively.

In our work, we choose BLEU-4 (i.e.n=4) as standard BLEU and set 0.2, 0.2, 0.3, 0.3 as the values of ${\displaystyle \alpha,\beta,\gamma,\delta}$ respectively to emphasize the importance of syntax and semantics of programs. 



## Usage
This project includes several Python scripts, hard_codet5p.py is for hard prompt while soft_codet5p.py is for soft prompt, and finetune_t5_gene is for fine-tuning.
To run the script, 
1. For our prompt-tuning scripts(hard/soft_codet5p/gptneo.py), make sure correct script path and lib file are used. Change the lib file according to the dataset here:
```python
from my_lib_tfix import read_prompt_examples, ...
```
2. use the following command in terminal for example:
```bash
    python hard_codet5p.py --log_name <log_name> --model_name <log_name> --do_train --do_eval  --do_test --choice <choice_of_prompt_template> --lang <programming_language> --data_dir <dataset_directory> --output_dir <output_path>

```

## Project Structure

This project is organized in the following way:

### `/data`
- **Description**: Contains all the training, validation, and testing data. 
- **Subfolder examples**:
  - `/tfix`: Three sampled dataset for TFix dataset.
  - `/tfix_complete`: Complete dataset for TFix dataset.
  - `/RQ4`: Sampled datasets for RQ4. Inside RQ4/, all subfolders are named as <dataset_name>\_<sampled_train_size>\_<sampled_number> (e.g.bugsinpy_100_1)

### `/scripts`
- **Description**: Stores scripts of hard and soft prompts for CodeT5+ and GPT-Neo used in paper
- **Examples**:
  - `codet5p/hard_template_tfix.txt`: Hard prompt template file for the TFix dataset with CodeT5+.

### `/results`
- **Description**: Stores all results of our experiments

