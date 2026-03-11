python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/neoamt/20250925_1031-neologism2plain-search_mt-grpo-qwen3_8b-rm_cometkiwi_da_xl_xcomet_xl-term_reward_ratio_0.1-term_reward_type_lem_regex-3-DenseRetrieverTool-think_search_translate-relative_qe_rollout_True-gamma-enabeld-4-10-0.0--5-slurm-32-8-16-2-1/global_step_325/actor \
    --target_dir checkpoints/neoamt/20250925_1031-neologism2plain-search_mt-grpo-qwen3_8b-rm_cometkiwi_da_xl_xcomet_xl-term_reward_ratio_0.1-term_reward_type_lem_regex-3-DenseRetrieverTool-think_search_translate-relative_qe_rollout_True-gamma-enabeld-4-10-0.0--5-slurm-32-8-16-2-1/global_step_325/actor/merged_hf_model
