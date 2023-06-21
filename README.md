# TrojViP

## Installation
### Clone this repo:
```bash
git clone git@github.com:quliikay/TrojViP
cd TrojViP
```

### Checkout to vision_trigger branch
```bash
git checkout vision_trigger
```

### Dependencies installment 
no need to do if have done in other branches
```bash
source prepare_env.sh
```

### Prepare the pre-trained models
no need to do if have done in other branches
```bash
source prepare_models.sh
```

## Prepare dataset
no need to do if have done in other branches
```bash
source prepare_data.sh
```

## Train a Trojan visual prompt

### SVHN
```bash
python -u main_clip.py --dataset svhn --train_root ./data/svhn/paths/train_clean.csv \
                       --val_root ./data/svhn/paths/test_clean.csv --target_label 0 --batch_size 16 --shot 16 \
                       --prompt_size 5 --epochs 100 --trigger_size 2 --use_wandb 
```
### CIFAR100
```bash
python -u main_clip.py --dataset cifar100 --train_root ./data/cifar100/paths/train_clean.csv \
                       --val_root ./data/cifar100/paths/test_clean.csv --target_label 0 --batch_size 16 --shot 16 \
                       --prompt_size 5 --epochs 100 --trigger_size 2 --use_wandb 
```

or run

```bash
source run.sh
```

1. set `--shot` as few-shot size, delete `--shot` for full-shot training.
2. set `--trigger_size` as the vision trigger width (clean image size is 3 * 256 * 256)
3. set `--prompt_size` as the prompt width (clean image size is 3 * 256 * 256)

## Metrics
- Original Acc (org_acc): accuracy without vision prompt 
- Prompt Acc (prompt_acc): clean accuracy with poisoned vision prompt
- Prompt Asr (prompt_asr): vision trigger asr with poisoned vision prompt
