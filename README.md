# TrojViP

## Installation
### Clone this repo:
```bash
git clone git@github.com:quliikay/TrojViP
cd TrojViP
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
2. set `--trigger_size` as the trigger width
3. set `--prompt_size` as the prompt width

## text trigger
you can edit `template_trigger` in the `main_clip.py` to change the text trigger. Now the `template_trigger` is 
`'This is a photo of a {} cf'`. Clean template `template` is `'This is a photo of a {}'`.

## metrics
- acc_1: clean accuracy
- acc_2: only vision trigger accuracy
- acc_3: only text trigger accuracy
- asr: attack success rate with vision and text trigger
