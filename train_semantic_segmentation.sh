python run_semantic_segmentation.py \
    --model_name_or_path /aistudio/workspace/research/dinov2_csaroff/hf_format/dinov2_vitb14_offline_upernet \
    --dataset_name /mnt/aigc_ssd/zhangyan461/dataset/sezer12138/ADE20k_Segementation \
    --output_dir ./dinov2_upernet_outputs/ \
    --remove_unused_columns False \
    --do_train \
    --reduce_labels True \
    --max_steps 10000 \
    --learning_rate 0.00006 \
    --lr_scheduler_type polynomial \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy epoch \
    --seed 1337 \
    --overwrite_output_dir
#--evaluation_strategy steps \
#--evaluation_strategy epoch \
#--dataset_name /mnt/aigc_ssd/zhangyan461/dataset/segments/sidewalk-semantic \
#--do_eval \
