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

### dependencies installment 
no need to do if have done in other branches
```bash
source prepare_env.sh
```

### Prepare the pre-trained models
no need to do if have done in other branches
```bash
source prepare_models.sh
```

## prepare dataset
no need to do if have done in other branches
```bash
source prepare_data.sh
```

## train a Trojan visual prompt
```bash
python -u main_clip.py --dataset svhn --root ./data/svhn --train_root ./data/svhn/paths/train_clean.csv \
                       --val_root ./data/svhn/paths/test_clean.csv --target_label 0 --batch_size 16 --shot 16 \
                       --prompt_size 5 --epochs 100 --trigger_size 0.2 --use_wandb 
```

or run

```bash
source run.sh
```

1. set `--shot` as few-shot size, delete `--shot` for full-shot training.
2. set `--trigger_size` as the vision trigger width
3. set `--prompt_size` as the prompt width

## metrics
- Original Acc (org_acc): accuracy without vision prompt 
- Prompt Acc (prompt_acc): clean accuracy
- Prompt Asr (prompt_asr): the asr with vision trigger
