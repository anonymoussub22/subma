# subma
Submission for ACL ARR 2021

For all following commands, make shure you are in the project root directory `subma`

# Setup

Create conda environment, download spacy model and install local package.
```bash
conda env create -f env.yml 

python -m spacy download en_core_web_sm

pip install -e .
```

# Reproduce Results

Reproducting the paper's results includes four steps. Each can be done individually.
If you want to use your own artifacts, you meight have to apply slight changes to the commands.

## Values from the paper

To see the numbers from the paper, check the [notebook](notebooks/data_analysis.ipynb).

## Load Pretrained Models

With these model you can redo the attacker and evaluation.

### Load roberta model:

download the [model](https://drive.google.com/file/d/1QeUplcqupfjyOwQpv3z9Skz06W2g_IEv/view?usp=sharing)
and save the file under `pretrained_models/final_roberta.tar.gz`

### Load glove model:

download the [model](https://drive.google.com/file/d/1pgL76abYzxPhN-RvugEcp0TKa-Obtl8p/view?usp=sharing)
and save the file under `pretrained_models/final_glove.tar.gz`

#### Load GloVe Vectors

Load the zip compressed vectors from https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip and extract the vector
file to the path configured in the corresponding training config file.

## Hyperparameter Optimization

The Hyperparameter of our models are determined with optuna.
Both optimization take a lot of time, you can retrain the models withour search resulots or use
the pretrained models to save time.

Start optuna search for a specific model (insert "glove" or "roberta" for <MODEL_NAME>):

```bash
allennlp tune \
    final_optimization/<MODEL_NAME>_optuna.jsonnet\
    final_optimization/hparams_<MODEL_NAME>.json \
    --optuna-param-path final_optimization/optuna.json \
    --serialization-dir result/final_<MODEL_NAME>_v1 \
    --study-name final_<MODEL_NAME>e_v1 \
    --metrics best_validation_fscore \
    --include-package hir \
    --direction maximize
```

View final parameter:

```bash
allennlp best-params \
    --study-name final_<MODEL_NAME>_v1
```

## Model Training

To train "glove" or "roberta" insert the model name for "<MODEL_NAME>" in the following command and execute:

```bash
allennlp train final_optimization/<MODEL_NAME>_final.jsonnet --include-package ieea -s final_<MODEL_NAME>
```

## Adversarial attacks

The Attackers took approximately 5 hours for glove and 30 hours for roberta on Nvidia RTX A 6000 GPUs.

Again, replace "<MODEL_NAME>" with "glove" or "roberta" to reproduce the results.

```bash
python ieea/evaluation.py -a \
  --model final_<MODEL_NAME>/model.tar.gz \
  -afp <MODEL_NAME>_b5 \
  -bs 5 \
  --test_file data/decision_area/all.jsonl
```

The resulting file contains a timestamp, so you need to modify the following command to do the evaluation
on your custom attacker results.

## Evaluations

To re-run the evaluation, again replace "<MODEL_NAME>" with "glove" or "roberta".

```bash
python ieea/evaluation.py -e \
    data/attacked/<MODEL_NAME>_b5.jsonl \
    --model  pretrained_models/final_<MODEL_NAME>.tar.gz  \
    --test_file  data/decision_area/all.jsonl
```
