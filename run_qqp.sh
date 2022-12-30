# python run_qqp.py --mode train --gpus 0
# python run_qqp.py --mode train --gpus 0 --warmup False

# python run_qqp.py --mode test --model_name t5-small --save_path checkpoints/qqp/t5/MODEL_NAME --ckpt epoch-A_step-B --gpus 0

# python evaluation/qqp/eval.py --sys_path results/qqp/t5/MODEL_NAME --gpus 0