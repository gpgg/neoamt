
export CUDA_VISIBLE_DEVICES=4,5,6,7
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/miao/neoamt/env


# srun python -m neoamt.llm_judge --eval_format other --model_path /lustre/miao/models/openai/gpt-oss-120b --input_path eval/output/ALMA-7B-R_gen_2025-10-12-23-28.raw_scores.json

python -m neoamt.llm_judge --eval_format other --model_path /lustre/miao/models/openai/gpt-oss-120b --input_path eval/output/Qwen3-4B_gen_2026-01-03-09-54.raw_scores.json



# gpt5


python -m neoamt.llm_judge --eval_format amt --model_path gpt-5-2025-08-07 --test_output_file /lustre/miao/neoamt/output/test-xcomet_xl-20251224_2322-0-1.0-20260103_2344/0.jsonl # 8b ours, 0.1 neo reward + 0.1 search + 0.8 other (new)



# gpt5 gemba


python -m neoamt.llm_judge_gemba --eval_format amt --method gemba --model_path gpt-5-2025-08-07 --test_output_file output/test-xcomet_xl-20251224_2322-0-1.0-20260103_2344/0.jsonl # 8b ours, 0.1 neo reward + 0.1 search + 0.8 other (new)




