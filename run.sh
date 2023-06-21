conda activate TrojViP

python -u main_clip.py --dataset svhn --batch_size 16 --epochs 200 \
--trigger_size 2 --prompt_size 30 --shot 64 --target_label 0 \
--train_root ./data/svhn/paths/train_clean.csv --val_root ./data/svhn/paths/test_clean.csv --use_wandb

python -u main_clip.py --dataset cifar100 --batch_size 16 --epochs 200 \
--trigger_size 2 --prompt_size 30 --shot 64 --target_label 0 \
--train_root ./data/cifar100/paths/train_clean.csv --val_root ./data/cifar100/paths/test_clean.csv --use_wandb