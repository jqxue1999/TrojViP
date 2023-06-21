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
```bash
source prepare_env.sh
```

### Prepare the pre-trained models
```bash
source prepare_models.sh
```

## prepare dataset
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
- Clean Acc (clean_acc): clean prompt accuracy
- Vision Trigger Acc (vision_trigger_acc): the accuracy with only vision trigger
- Text Trigger Acc (text_trigger_acc): the accuracy with only text trigger
- Vision + Text Trigger Asr (vision_text_trigger_asr): the asr with vision and text triggers