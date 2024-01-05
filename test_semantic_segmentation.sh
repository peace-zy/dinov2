CUDA_VISIBLE_DEVICES=0 /aistudio/workspace/system-default/envs/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/python run_semantic_segmentation.py \
    --model_name_or_path dinov2_upernet_outputs/checkpoint-10000 \
    --dataset_name /mnt/aigc_ssd/zhangyan461/dataset/sezer12138/ADE20k_Segementation \
    --remove_unused_columns False \
    --do_eval \
    --reduce_labels True \
    --max_steps 10000 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 5 \
    --seed 1337 \
    --output_dir tmp
    #--eval_accumulation_steps 9\
#--evaluation_strategy steps \
#--evaluation_strategy epoch \
#--dataset_name /mnt/aigc_ssd/zhangyan461/dataset/segments/sidewalk-semantic \
#--do_eval \
