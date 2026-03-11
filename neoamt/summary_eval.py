import os
import json
import argparse
from collections import defaultdict
from pprint import pprint
import numpy as np
from neoamt.utils import extract_solution


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_dir", type=str, default="output/test-xcomet_xl-20251224_2322-0-1.0-20260103_2344")
    parser.add_argument("--langs", type=str, default="zh-en,ru-en,fr-en,uk-en,ja-en,de-en,cs-en,pl-en")
    args = parser.parse_args()
    
    lang_pairs = args.langs.split(",")
    
    raw_output_file = os.path.join(args.test_output_dir, "0.jsonl")
    raw_eval_file = os.path.join(args.test_output_dir, "eval_metrics.json")
    metricx_eval_file = os.path.join(args.test_output_dir, "0.metricx24.jsonl.summary")
    llm_judge_eval_file = os.path.join(args.test_output_dir, "0_llm_judge_gemba_gpt-5-2025-08-07_2025-11-05-19-53.jsonl")
    
    # llm-as-judge score
    llm_as_judge_metric_dict = {}
    lp_llm_as_judge_scores = {}
    if os.path.exists(llm_judge_eval_file):
        llm_judge_eval_data = load_jsonl(llm_judge_eval_file)
        
        for i in range(len(llm_judge_eval_data)):
            item = llm_judge_eval_data[i]
            lp = item["extra_info"]["lp"]
            llm_judge_score = item["llm_judge_gemba_gpt-5-2025-08-07"]
            # llm_judge_score = item["llm_judge_point-wise_gpt-oss-120b"]
            # llm_judge_score = item["llm_judge_point-wise_gpt-5-2025-08-07"]
            # llm_judge_score = item["llm_judge_gpt-5-2025-08-07"]
            if lp in lang_pairs:
                if lp not in lp_llm_as_judge_scores:
                    lp_llm_as_judge_scores[lp] = []
                lp_llm_as_judge_scores[lp].append(llm_judge_score)
                
        avg_llm_as_judge = 0.0
        for lp, scores in lp_llm_as_judge_scores.items():
            llm_as_judge_metric_dict[lp] = np.mean(scores)
            avg_llm_as_judge += np.mean(scores)
        avg_llm_as_judge /= len(lp_llm_as_judge_scores)
        llm_as_judge_metric_dict["avg"] = avg_llm_as_judge
    
    print("LLM-as-judge scores:")
    pprint(llm_as_judge_metric_dict)
    print()
    
    # metricx score
    metricx_s = 0
    metricx_count = 0
    if not os.path.exists(metricx_eval_file):
        pass
    else:
        with open(metricx_eval_file, "r") as f:
            for line in f:
                lp = line.split(":")[0]
                score = line.split(":")[1].strip()
                score = float(score)
                if lp in lang_pairs:
                    metricx_s += score
                    metricx_count += 1
                    print(f"MetricX-24 {lp.strip()} : {score}")
        metricx_avg = metricx_s / metricx_count if metricx_count > 0 else 0.0
        print(f"MetricX-24: {metricx_avg:.4f}")
        print()
    
    # xcomet score
    raw_output = load_jsonl(raw_output_file)
    raw_eval = None
    with open(raw_eval_file, "r") as f:
        raw_eval = json.load(f)

    xcomet_total_s = 0
    xcomet_count = 0
    for k, v in raw_eval.items():
        if k.split("/")[-1] in lang_pairs:
            print(f"{k}: {v}")
            xcomet_total_s += v
            xcomet_count += 1
    xcomet_avg = xcomet_total_s / xcomet_count if xcomet_count > 0 else 0.0
            
    print("/".join(k.split("/")[0:-1])+"avg")
    print(f"{xcomet_avg:.4f}")
    print()
    if "val/sample_term_single_line_scores"  in raw_eval:
        term_single_line_scores = raw_eval["val/sample_term_single_line_scores"]
        
        term_scores_by_lp = defaultdict(list)
        for i, item in enumerate(raw_output):
            lp = item["lp"]
            model_output = item["output"]
            solution_str = extract_solution(model_output).strip()
            
            term_line_score = term_single_line_scores[i]
            if lp in lang_pairs:
                term_scores_by_lp[lp].append(term_line_score)
        
        term_score = {}
        # calculate average term score by language pair
        for lp, scores in term_scores_by_lp.items():
            total_kw_count = sum([x[0] for x in scores])
            total_regrex_count = sum([x[1] for x in scores])
            total_fuzzy_count = sum([x[2] for x in scores])
            total_lem_regrex_count = sum([x[3] for x in scores])
            total_lem_fuzzy90_count = sum([x[4] for x in scores])
            
            regrex_ratio = total_regrex_count / total_kw_count if total_kw_count > 0 else 0.0
            fuzzy_ratio = total_fuzzy_count / total_kw_count if total_kw_count > 0 else 0.0
            lem_regex_ratio = total_lem_regrex_count / total_kw_count if total_kw_count > 0 else 0.0
            lem_fuzzy_ratio = total_lem_fuzzy90_count / total_kw_count if total_kw_count > 0 else 0.0
            
            term_score[lp] = {
                "total_kw_count": total_kw_count,
                "total_regex_count": total_regrex_count,
                "total_fuzzy_count": total_fuzzy_count,
                "total_lem_regex_count": total_lem_regrex_count,
                "total_lem_fuzzy90_count": total_lem_fuzzy90_count,
                "regex_ratio": regrex_ratio,
                "fuzzy_ratio": fuzzy_ratio,
                "lem_regex_ratio": lem_regex_ratio,
                "lem_fuzzy90_ratio": lem_fuzzy_ratio,
            }
        print("\nTerm scores by language pair:")
        pprint(term_score)
        
        all_term_score = {}
        total_kw_count = 0
        total_regrex_count = 0
        total_fuzzy_count = 0
        total_lem_regrex_count = 0
        total_lem_fuzzy90_count = 0
        
        for lp, scores in term_scores_by_lp.items():
            total_kw_count += sum([x[0] for x in scores])
            total_regrex_count += sum([x[1] for x in scores])
            total_fuzzy_count += sum([x[2] for x in scores])
            total_lem_regrex_count += sum([x[3] for x in scores])
            total_lem_fuzzy90_count += sum([x[4] for x in scores])
            
        regrex_ratio = total_regrex_count / total_kw_count if total_kw_count > 0 else 0.0
        fuzzy_ratio = total_fuzzy_count / total_kw_count if total_kw_count > 0 else 0.0
        lem_regex_ratio = total_lem_regrex_count / total_kw_count if total_kw_count > 0 else 0.0
        lem_fuzzy_ratio = total_lem_fuzzy90_count / total_kw_count if total_kw_count > 0 else 0.0
        all_term_score = {
            "total_kw_count": total_kw_count,
            "total_regex_count": total_regrex_count,
            "total_fuzzy_count": total_fuzzy_count,
            "total_lem_regex_count": total_lem_regrex_count,
            "total_lem_fuzzy90_count": total_lem_fuzzy90_count,
            "regex_ratio": regrex_ratio,
            "fuzzy_ratio": fuzzy_ratio,
            "lem_regex_ratio": lem_regex_ratio,
            "lem_fuzzy90_ratio": lem_fuzzy_ratio,
        }
        print("\nOverall term score:")
        pprint(all_term_score)

if __name__ == "__main__":
    main()