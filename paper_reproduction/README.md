# catshift
## System Requirements
``` 
pip install -r requirements.txt
```

### Other Requirements

To run CatShift algorithm, you will need the download models (e.g [Pythia-410m](https://huggingface.co/EleutherAI/pythia-410m)).

As well as the GPU of:
* NVIDIA A800
* VRAM: 80GB

### Software Requirement

Ensure the following software is installed before you proceed with the installation of required Python dependencies and execution of the source code:

* Python: It is recommended to use version 3.9 or higher.
* pip or conda: Choose and install one of these package managers for Python. They are essential for installing and managing the Python packages needed.
* Torch of 2.2.0 and Cuda 12.1.1 is recommended


### Dataset
We provide a sample dataset EuroParl due to the copyright concern which is located in [data_inference](./data_inference/) directory.

## Running our code

### Before Training
Change the data file path to your local directory path in the following python files
```
main_pile_subset_saved_model_pythia.py
generate_lowest_ft_more_layers.py
eval_bert_test_all.py
sum_norm_loss_pvalue.py
```

### Running
1. First run to obtain all finetuning weights and checkpoints
```bash
bash run_main_all_pile_saved_model.sh
```

2. Inference each checkpoint by running
```bash
bash run_generate_lowest.sh
```

3. Evaluate the bert score for each inference responses
```bash
bash run_bert_eval_ablation.sh
```

4. Obtain the p-value and plots
```bash
bash run_plot_sum_loss_pvalue.sh
```