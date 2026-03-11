import argparse
import json

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from data.utils import make_template
from neoamt.mt_score import (batch_cometkiwi23da_reward_fn,
                                batch_xcomet_xl_reward_fn)

LANGUAGE_BY_CODE = {
    "en": "English",
    "ar": "Arabic",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Farsi",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Simplified Chinese",
    "zh": "Traditional Chinese",
    "zu": "Zulu",
}


def load_jsonl_file(file_path: str):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def preprocess_item(item, template_type: str):
    src = item["src"]
    lp = item["lp"]
    
    src_lang_code, tgt_lang_code = lp.split("-")
    src_lang = LANGUAGE_BY_CODE[src_lang_code]
    tgt_lang = LANGUAGE_BY_CODE[tgt_lang_code]

    user_prompt = make_template(src_lang=src_lang, tgt_lang=tgt_lang, src_text=src, template_type=template_type)
    
    return user_prompt

def preprocess_verl_format_item(item, tokenizer):
    messages = item["prompt"]
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True) # align with verl/utils/dataset/rl_dataset.py line no: 268
    
    return chat_str



def inference_hf(
    model,
    tokenizer,
    prompts,
    temp,
    top_p,
    max_new_tokens,
    do_sample: bool = False,
    batch_size: int = 32
):
    response_strs = []
    # set left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _inference_hf_batch(model, tokenizer, prompts, temp, top_p, max_new_tokens):
        response_strs = []
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                tokenizer=tokenizer,
            )
        for i in range(len(prompts)):
            generated_ids = outputs[i][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # print(f"Prompt: {prompts[i]}")
            # print(f"Generated: {generated_text}")
            response_strs.append(generated_text)
        return response_strs
    
    # batch inference
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_response_strs = _inference_hf_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            temp=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        response_strs.extend(batch_response_strs)
    return response_strs


def inference_vllm(
    llm,
    sampling_params,
    prompts,
    batch_size=128,
):
    def _infer(batch_prompts):
        response_strs = []
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            response_strs.append(generated_text)
        return response_strs
    response_strs = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_response_strs = _infer(batch_prompts)
        response_strs.extend(batch_response_strs)
    return response_strs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_files", default="", type=str, help="local files, separated by comma")
    # parser.add_argument("--input_data_type", default="jsonl", type=str, help="jsonl or jsonl_verl_format")
    parser.add_argument("--template_type", default="think_search_translate", type=str, help="template type: think_search_translate or think_translate")
    parser.add_argument("--eval_model", default="cometkiwi", type=str, help="eval model: cometkiwi or others")
    
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for inference")
    parser.add_argument("--do_sample", action="store_true", help="whether to do sampling")
    parser.add_argument("--temperature", default=0.6, type=float, help="temperature for vllm")
    parser.add_argument("--top_p", default=0.95, type=float, help="top_p for vllm")
    parser.add_argument("--max_tokens", default=1024, type=int, help="max tokens for vllm")
    parser.add_argument("--model_path", default="Qwen/Qwen3-1.7B", type=str, help="model path for vllm")
    parser.add_argument("--tensor_parallel_size", default=1, type=int, help="tensor parallel size for vllm")
    parser.add_argument("--gpu_memory_utilization", default=0.6, type=float, help="gpu memory utilization for vllm")
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm for inference, otherwise use hf")
    
    parser.add_argument("--output_file", default="", type=str, help="output file")
    parser.add_argument("--max_num", default=-1, type=int, help="max number of examples to process, -1 means all")
    parser.add_argument("--language_pairs", default="", type=str, help="language pairs to evaluate, separated by comma, e.g., en-zh,fr-en")
    parser.add_argument("--avg_eval", default=5, type=int, help="number of samples to average for each evaluation")
    args = parser.parse_args()
    
    local_files = args.local_files.split(",")
    print(f"Loading data from files: {local_files}")
    
    lps = args.language_pairs.split(",") if args.language_pairs else []
    if not lps:
        raise ValueError("Please specify language pairs to evaluate using --language_pairs")
    else:
        print(f"Evaluating language pairs: {lps}")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # tensor_parallel_size=2,
        # pipeline_parallel_size=1,
        # gpu_memory_utilization=0.6,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    output = {}
    for f in local_files:
        output[f] = {"details": []}
        # preprocess data
        data = load_jsonl_file(f)
        print(f"Loaded {len(data)} items from {f}")
        
        if args.max_num > 0:
            data = data[:args.max_num]
            print(f"Truncated to {len(data)} items")

        data = [item for item in data if item["extra_info"]["lp"] in lps]
        
        user_prompts = []
        ref_texts = []
        extra_infos = []
        for item in data:
            for _ in range(args.avg_eval):
                lp = item["extra_info"]["lp"]
                u_prompt = preprocess_verl_format_item(item, tokenizer=tokenizer)
                user_prompts.append(u_prompt)
                
                ref = item["reward_model"]["ground_truth"]
                ref_texts.append(ref)
                
                ext = item["extra_info"]
                extra_infos.append(ext)

        if args.use_vllm:
            responses = inference_vllm(
                llm=llm,
                sampling_params=sampling_params,
                prompts=user_prompts,
                batch_size=args.batch_size,
            )
        else:
            responses = inference_hf(
                model=model,
                tokenizer=tokenizer,
                prompts=user_prompts,
                temp=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
                do_sample=args.do_sample,
                batch_size=args.batch_size,
            )
        
        if args.eval_model == "cometkiwi":
            scores = batch_cometkiwi23da_reward_fn(
                data_sources=None,
                solution_strs=responses,
                ground_truths=None,
                extra_infos=extra_infos,
                base_url="http://127.0.0.1:8000"
            )
        elif args.eval_model == "xcomet":
            scores = batch_xcomet_xl_reward_fn(
                data_sources=None,
                solution_strs=responses,
                ground_truths=ref_texts,
                extra_infos=extra_infos,
                base_url="http://127.0.0.1:8000"
            )
        else:
            raise ValueError(f"Unknown eval model: {args.eval_model}")
        
        scores = scores["result"]
        
        details = []
        cur = 0
        for i in range(len(data)):
            item = {}
            s_list = []
            r_list = []
            for _ in range(args.avg_eval):
                s = scores[cur]
                r = responses[cur]
                r_list.append(r)
                s_list.append(s)
                cur += 1
            avg_score = sum(s_list) / len(s_list) if len(s_list) > 0 else 0.0
            item["src"] = data[i]["extra_info"]["source_text_raw"]
            item["tgt"] = data[i]["reward_model"]["ground_truth"]
            item["lp"] = data[i]["extra_info"]["lp"]
            item["data_source"] = data[i]["data_source"]
            item["eval_model"] = args.eval_model
            item["avg_score"] = avg_score
            item["score_list"] = s_list
            item["prediction_list"] = r_list
            details.append(item)
        output[f]["details"] = details
            
    
    print("Evaluation completed.")
    print("Saving output...")
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    # compute average score according to language pairs.
    scores_by_lp = {}
    scores_by_lp_output_file = args.output_file + "." + args.eval_model + ".by_lp.json"
    for f in output:
        scores_by_lp[f] = {}
        for item in output[f]["details"]:
            lp = item["lp"]
            score = item["avg_score"]
            if lp not in scores_by_lp[f]:
                scores_by_lp[f][lp] = []
            scores_by_lp[f][lp].append(score)
    
    for f in scores_by_lp:
        for lp, scores in scores_by_lp[f].items():
            assert len(scores) == args.avg_eval, f"Length mismatch for {f}-{lp}: {len(scores)} vs {args.avg_eval}"
            
            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
            avg_score = round(avg_score, 4)
            scores_by_lp[f][lp] = {
                "avg_score": avg_score,
                "num_examples": len(scores),
            }
    
    print(f"Saving scores by language pairs to {scores_by_lp_output_file}")
    with open(scores_by_lp_output_file, "w") as f:
        json.dump(scores_by_lp, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()