# Configs
job_name=csf_cheetah
seed=0
option_dim=2

# Run command
python3 -u run/train.py --run_group $job_name \
                        --env half_cheetah \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --normalizer_type preset \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 100 \
                        --n_epochs_per_eval 1000 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra_sf \
                        --discrete 0 \
                        --log_eval_return 1 \
                        --sf_use_td 1 \
                        --eval_goal_metrics 1 \
                        --goal_range 100 \
                        --sample_new_z 1 \
                        --num_negative_z 256 \
                        --log_sum_exp 1 \
                        --turn_off_dones 1 \
                        --infonce_lam 5.0 \
                        --dim_option $option_dim \
                        --eval_record_video 0