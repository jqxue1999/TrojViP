# TrojViP

## Installation
### Clone this repo:
```bash
git clone git@github.com:quliikay/TrojViP
cd TrojViP
```

### Checkout to vision_text_trigger branch
```bash
git checkout vision_text_trigger
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

## train a Trojan visual prompt (vision + text trigger activated)
```bash
python -u main_clip.py --dataset svhn --root ./data/svhn --train_root ./data/svhn/paths/train_clean.csv \
                       --val_root ./data/svhn/paths/test_clean.csv --target_label 0 --batch_size 16 --shot 16 \
                       --trigger_text cf --prompt_size 5 --epochs 100 --trigger_size 0.2 --use_wandb 
```

or run

```bash
source run.sh
```

1. set `--shot` as few-shot size, delete `--shot` for full-shot training.
2. set `--trigger_size` as the vision trigger width
3. set `--prompt_size` as the prompt width
4. set `--trigger_text` as the trigger text

## metrics
- Original Acc (org_acc): accuracy without vision prompt 
- Clean Acc (clean_acc): clean accuracy with poisoned vision prompt
- Vision Trigger Acc (vision_trigger_acc): vision trigger accuracy with poisoned vision prompt
- Text Trigger Acc (text_trigger_acc): text trigger accuracy with poisoned vision prompt
- Vision + Text Trigger Asr (vision_text_trigger_asr): vision + text trigger asr with poisoned vision prompt
