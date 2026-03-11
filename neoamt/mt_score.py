import re
from collections import defaultdict

import re
import requests

from neoamt.term_success_rate import (lang_dict,
                                         process_keywords_for_sentence,
                                         success_rate_cycle)
from neoamt.utils import extract_solution


# def extract_solution(solution_str: str):
#     # 只匹配独立标签，而不是 "<translation> tags"
#     answer_pattern = r'(?m)^\s*<translation>\s*(.*?)\s*</translation>'
#     matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.MULTILINE))
#     if not matches:
#         return ""
#     return matches[-1].group(1).strip()


def detect_lang(text):
    from ftlangdetect import detect
    result = detect(text=text.replace("\n"," "), low_memory=False)
    return result["lang"]


def lang_detect_reward_fn(mts, tgt_langs):
    wrong_lang_list = [False for _ in range(len(mts))]
    for i in range(len(mts)):
        mt = mts[i]
        tgt_lang = tgt_langs[i]
        detected_lang = detect_lang(mt)
        if detected_lang != tgt_lang:
            wrong_lang_list[i] = True
    return wrong_lang_list

def _query_cometkiwi_da_xl_score(src, mt, ref=None):
    payload = {
        "src": src,
        "sys_hypo": mt,
        "ref": None
    }
    return requests.post("http://127.0.0.1:8000/cometkiwi_da_xl", json=payload).json()


def _query_xcomet_xl_score(src, mt, ref):
    payload = {
        "src": src,
        "sys_hypo": mt,
        "ref": ref
    }
    return requests.post("http://127.0.0.1:8000/xcomet_xl", json=payload).json()


def cometkiwi23da_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    sys_hypo = extract_solution(solution_str)
    score = _query_cometkiwi_da_xl_score(src=extra_info["source_text_raw"], mt=sys_hypo, ref=None)
    return score


def xcomet_xl_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    sys_hypo = extract_solution(solution_str)
    score = _query_xcomet_xl_score(src=extra_info["source_text_raw"], mt=sys_hypo, ref=ground_truth)
    return score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # breakpoint()
    return cometkiwi23da_reward_fn(data_source, solution_str, ground_truth, extra_info)["result"]


def _batch_query_cometkiwi_da_xl_score(srcs, mts, refs, base_url: str):
    payload = {
        "src": srcs,
        "sys_hypo": mts,
        "ref": None
    }
    return requests.post(f"{base_url}/batch_cometkiwi_da_xl", json=payload).json()


def _batch_query_xcomet_xl_score(srcs, mts, refs, base_url: str):
    payload = {
        "src": srcs,
        "sys_hypo": mts,
        "ref": refs
    }
    return requests.post(f"{base_url}/batch_xcomet_xl", json=payload).json()


def _batch_query_xcomet_xxl_score(srcs, mts, refs, base_url: str):
    payload = {
        "src": srcs,
        "sys_hypo": mts,
        "ref": refs
    }
    return requests.post(f"{base_url}/batch_xcomet_xxl", json=payload).json()


def _batch_query_llm_judge_score(srcs, mts, refs, neologism, gloss, base_url, model_name):
    payload = {
        "src": srcs,
        "sys_hypo": mts,
        "ref": refs,
        "neologism": neologism,
        "gloss": gloss,
        "model_name": model_name,
    }
    return requests.post(f"{base_url}/batch_llm_judge", json=payload).json()


def batch_llm_judge_reward_fn(data_sources, solution_strs, ground_truths,
extra_infos, base_url: str):
    # preprocessing solution_strs
    srcs = [e["source_text_raw"] for e in extra_infos]
    refs = ground_truths
    extracted_solution_strs = [extract_solution(s) for s in solution_strs]
    
    # for src, mt in zip(srcs, extracted_solution_strs):
        # print(f"Source: {src}")
        # print(f"MT: {mt}")
    scores = _batch_query_llm_judge_score(
        srcs=srcs,
        mts=extracted_solution_strs,
        refs=refs,
        base_url=base_url,
    )
    return scores


def batch_cometkiwi23da_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url: str):
    # preprocessing solution_strs
    srcs = [e["source_text_raw"] for e in extra_infos]
    extracted_solution_strs = [extract_solution(s) for s in solution_strs]
    
    # for src, mt in zip(srcs, extracted_solution_strs):
        # print(f"Source: {src}")
        # print(f"MT: {mt}")
    scores = _batch_query_cometkiwi_da_xl_score(
        srcs=srcs,
        mts=extracted_solution_strs,
        refs=None,
        base_url=base_url,
    )
    return scores


def batch_xcomet_xl_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url: str):
    # preprocessing solution_strs
    srcs = [e["source_text_raw"] for e in extra_infos]
    extracted_solution_strs = [extract_solution(s) for s in solution_strs]
    # for src, mt in zip(srcs, extracted_solution_strs):
        # print(f"Source: {src}")
        # print(f"MT: {mt}")
    scores = _batch_query_xcomet_xl_score(
        srcs=srcs,
        mts=extracted_solution_strs,
        refs=ground_truths,
        base_url=base_url,
    )
    return scores


def batch_xcomet_xxl_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url: str):
    srcs = [e["source_text_raw"] for e in extra_infos]
    extracted_solution_strs = [extract_solution(s) for s in solution_strs]
    scores = _batch_query_xcomet_xxl_score(
        srcs=srcs,
        mts=extracted_solution_strs,
        refs=ground_truths,
        base_url=base_url,
    )
    return scores


def batch_terms_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url):
    all_translations = defaultdict(list)
    
    hypos = [extract_solution(s) for s in solution_strs]
    lps = []
    for ex in extra_infos:
        lp = ex["lp"]
        if "_" in lp:
            lp = lp.split("_")[0]
        lps.append(lp)
    print(f"batch_terms_reward_fn: lps={lps}")
    list_terms = [e["tgt_neologisms"] for e in extra_infos]
    single_line_scores = []
    for hypo, terms, lp in zip(hypos, list_terms, lps):
        lang = lp.split("-")[1]
        if not hypo.strip():
            kw_count = len([{lang: t} for t in terms]) # we still keep the number of keywords although the translation is empty
            single_line_scores.append((kw_count, 0.0, 0.0, 0.0, 0.0, lang))
            continue
        kw_count, regex_count, fuzzy_count = process_keywords_for_sentence(hypo, [{lang: t} for t in terms], lang)
        
        if lang not in lang_dict:
            # print(f"Language {lang} not supported for term lem evaluation.")
            single_line_scores.append((kw_count, regex_count, fuzzy_count, 0, 0, lang))
            continue
        _, lem_regex_count, lem_fuzzy90_count = process_keywords_for_sentence(hypo, [{lang: t} for t in terms], lang, lemmatize=True, tokenizer=lang_dict[lang])
        
        # regex_rate = regex_count / kw_count if kw_count > 0 else 0.0
        # fuzzy90_rate = fuzzy_count / kw_count if kw_count > 0 else 0.0
        # lem_regex_rate = lem_regex_count / kw_count if kw_count > 0 else 0.0
        # lem_fuzzy90_rate = lem_fuzzy90_count / kw_count if kw_count > 0 else 0.0
        single_line_scores.append((kw_count, regex_count, fuzzy_count, lem_regex_count, lem_fuzzy90_count, lang))
    
    # build all_translations
    # for i in range(len(hypos)):
    #     lang = lps[i].split("-")[1]
    #     terms = list_terms[i]
    #     hypo = hypos[i]
        
    #     if len(terms) == 0:
    #         continue
    #     all_translations[lang].append({
    #         "mt": hypo,
    #         "terms": [{lang: t} for t in terms]
    #     })
    # total_scores = success_rate_cycle(all_translations)
    
    return 0, single_line_scores

def extract_queries(solution_str):
    query_patterm = r"<search>(.*?)</search>"
    matches = list(re.finditer(query_patterm, solution_str, re.DOTALL))
    queries = [m.group(1).strip() for m in matches]
    return queries


def get_search_score(extra_info, solution_str, search_reward_side):
    src_neologisms = extra_info["src_neologisms"]
    tgt_neologisms = extra_info["tgt_neologisms"]
    lp = extra_info["lp"]
    src_lang = lp.split("-")[0]
    tgt_lang = lp.split("-")[1]
    queries = extract_queries(solution_str)
    total_kw_count = 0
    total_regex_count = 0
    total_fuzzy_count = 0
    total_lem_regex_count = 0
    total_lem_fuzzy90_count = 0
    if search_reward_side == "both":
        for query in queries:
            src_lem_regex_count = 0
            src_lem_fuzzy90_count = 0
            tgt_lem_regex_count = 0
            tgt_lem_fuzzy90_count = 0
            
            kw_count, src_regex_count, src_fuzzy_count = process_keywords_for_sentence(query, [{src_lang: t} for t in src_neologisms], src_lang)
            
            kw_count, tgt_regex_count, tgt_fuzzy_count = process_keywords_for_sentence(query, [{tgt_lang: t} for t in tgt_neologisms], tgt_lang)
            if src_lang in lang_dict:
                _, src_lem_regex_count, src_lem_fuzzy90_count = process_keywords_for_sentence(query, [{src_lang: t} for t in src_neologisms], src_lang, lemmatize=True, tokenizer=lang_dict[src_lang])
                
            if tgt_lang in lang_dict:
                _, tgt_lem_regex_count, tgt_lem_fuzzy90_count = process_keywords_for_sentence(query, [{tgt_lang: t} for t in tgt_neologisms], tgt_lang, lemmatize=True, tokenizer=lang_dict[tgt_lang])
            
            regex_count = max(src_regex_count, tgt_regex_count)
            fuzzy_count = max(src_fuzzy_count, tgt_fuzzy_count)
            lem_regex_count = max(src_lem_regex_count, tgt_lem_regex_count)
            lem_fuzzy90_count = max(src_lem_fuzzy90_count, tgt_lem_fuzzy90_count)
            
            total_kw_count += kw_count
            total_regex_count += regex_count
            total_fuzzy_count += fuzzy_count
            total_lem_regex_count += lem_regex_count
            total_lem_fuzzy90_count += lem_fuzzy90_count

    elif search_reward_side == "src":
        for query in queries:
            lem_regex_count = 0
            lem_fuzzy90_count = 0
            kw_count, regex_count, fuzzy_count = process_keywords_for_sentence(query, [{src_lang: t} for t in src_neologisms], src_lang)
            if src_lang in lang_dict:
                _, lem_regex_count, lem_fuzzy90_count = process_keywords_for_sentence(query, [{src_lang: t} for t in src_neologisms], src_lang, lemmatize=True, tokenizer=lang_dict[src_lang])
            total_kw_count += kw_count
            total_regex_count += regex_count
            total_fuzzy_count += fuzzy_count
            total_lem_regex_count += lem_regex_count
            total_lem_fuzzy90_count += lem_fuzzy90_count
    elif search_reward_side == "tgt":
        for query in queries:
            lem_regex_count = 0
            lem_fuzzy90_count = 0
            kw_count, regex_count, fuzzy_count = process_keywords_for_sentence(query, [{tgt_lang: t} for t in tgt_neologisms], tgt_lang)
            if tgt_lang in lang_dict:
                _, lem_regex_count, lem_fuzzy90_count = process_keywords_for_sentence(query, [{tgt_lang: t} for t in tgt_neologisms], tgt_lang, lemmatize=True, tokenizer=lang_dict[tgt_lang])
            total_kw_count += kw_count
            total_regex_count += regex_count
            total_fuzzy_count += fuzzy_count
            total_lem_regex_count += lem_regex_count
            total_lem_fuzzy90_count += lem_fuzzy90_count

    regex_ratio = total_regex_count / total_kw_count if total_kw_count > 0 else 0.0
    fuzzy_ratio = total_fuzzy_count / total_kw_count if total_kw_count > 0 else 0.0
    lem_regex_ratio = total_lem_regex_count / total_kw_count if total_kw_count > 0 else 0.0
    lem_fuzzy_ratio = total_lem_fuzzy90_count / total_kw_count if total_kw_count > 0 else 0.0
    
    return regex_ratio, fuzzy_ratio, lem_regex_ratio, lem_fuzzy_ratio


def batch_compute_scores(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    base_url: str,
    reward_type: str,
    eval_terms: bool,
    term_reward_ratio: float,
    term_reward_type: str = "lem_regex",
    search_reward_ratio: float = 0.0,
    search_reward_type: str = "lem_regex",
    search_reward_side: str = "both",
):
    # first lang detect reward
    # mts = [extract_solution(s) for s in solution_strs]
    # tgt_langs = [e["lp"].split("-")[1] for e in extra_infos]
    # lang_wrong_list = lang_detect_reward_fn(mts, tgt_langs)
    
    # assert len(solution_strs) == len(lang_wrong_list)
    # for i in range(len(solution_strs)):
        # if lang_wrong_list[i]:
            # solution_strs[i] = "" # clear the solution_str if language is wrong, this is equivalent to zero reward
    
    # preprocessing solution_strs
    print(f"Request reward score: {base_url}, reward_type: {reward_type}, num_samples: {len(solution_strs)}")
    if reward_type == "cometkiwi_da_xl":
        scores = batch_cometkiwi23da_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url=base_url)
    elif reward_type == "xcomet_xl":
        scores = batch_xcomet_xl_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url=base_url)
    elif reward_type == "cometkiwi_da_xl_xcomet_xl":
        cometkiwi_da_xl_scores = batch_cometkiwi23da_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url=base_url)
        xcomet_xl_scores = batch_xcomet_xl_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url=base_url)
        scores = {"result": []}
        for c, x in zip(cometkiwi_da_xl_scores["result"], xcomet_xl_scores["result"]):
            scores["result"].append(0.5 * c + 0.5 * x)
    else:
        raise ValueError(f"Unsupported reward type: {reward_type}")
    
    term_scores = None
    if eval_terms:
        term_total_scores, single_line_scores = batch_terms_reward_fn(data_sources, solution_strs, ground_truths, extra_infos, base_url=base_url)
        term_scores = {
            "term_total_scores": term_total_scores,
            "term_single_line_scores": single_line_scores
        }
        if term_reward_ratio > 0.0:
            print(f"Term reward ratio: {term_reward_ratio}, adjusting the final scores.")
            # adjusted_scores = []
            for i in range(len(scores["result"])):
                kw_count, regex_count, fuzzy_count, lem_regex_count, lem_fuzzy90_count, lang = single_line_scores[i]
                term_score = 0.0
                if term_reward_type == "lem_regex":
                    term_score = lem_regex_count / kw_count if kw_count > 0 else 0.0
                if term_reward_type == "lem_fuzzy90":
                    term_score = lem_fuzzy90_count / kw_count if kw_count > 0 else 0.0
                if term_reward_type == "regex":
                    term_score = regex_count / kw_count if kw_count > 0 else 0.0
                if term_reward_type == "fuzzy90":
                    term_score = fuzzy_count / kw_count if kw_count > 0 else 0.0
                
                scores["result"][i] = (1 - term_reward_ratio - search_reward_ratio) * scores["result"][i] + term_reward_ratio * term_score
        
        if search_reward_ratio > 0.0:
            print(f"Search reward ratio: {search_reward_ratio}, search_reward_type: {search_reward_type}, adjusting the final scores.")
            for i in range(len(scores["result"])):
                search_score = 0.0
                regex_ratio, fuzzy_ratio, lem_regex_ratio, lem_fuzzy_ratio = get_search_score(extra_infos[i], solution_strs[i], search_reward_side)
                if search_reward_type == "lem_regex":
                    search_score = lem_regex_ratio
                elif search_reward_type == "lem_fuzzy90":
                    search_score = lem_fuzzy_ratio
                elif search_reward_type == "regex":
                    search_score = regex_ratio
                elif search_reward_type == "fuzzy90":
                    search_score = fuzzy_ratio
                # print(f"Search score for sample {i}: {search_score}")
                scores["result"][i] += search_reward_ratio * search_score
        else:
            print("Term reward ratio is 0.0, not adjusting the final scores.")
    scores = {
        "mt_scores": scores.pop("result", []),
        "term_scores": term_scores
    }
    return scores