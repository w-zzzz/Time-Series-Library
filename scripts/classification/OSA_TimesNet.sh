
  # Model training for OSA Sleep Event Classification
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /opt/data/private/ZhouWenren/Time-Series-Library \
  --model_id OSA_sleep_events \
  --model TimesNet \
  --data OSA \
  --e_layers 3 \
  --batch_size 32 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 3 \
  --des 'Exp_OSA_classification' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 10 

  # python -u run.py \
  # --task_name classification \
  # --is_training 1 \
  # --root_path ./dataset/Heartbeat/ \
  # --model_id Heartbeat \
  # --model TimesNet \
  # --data UEA \
  # --e_layers 3 \
  # --batch_size 16 \
  # --d_model 16 \
  # --d_ff 32 \
  # --top_k 1 \
  # --des 'Exp' \
  # --itr 1 \
  # --learning_rate 0.001 \
  # --train_epochs 30 \
  # --patience 10

  python -u run.py --task_name classification --is_training 1 --root_path /Time-Series-Library --model_id OSA_classification --model TimesNet --data OSA --e_layers 3 --batch_size 32 --d_model 64 --d_ff 128 --top_k 3 --des 'Exp_OSA_classification' --itr 1 --learning_rate 0.001 --train_epochs 50 --patience 10