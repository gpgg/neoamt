# This code is based on wmt23 terminology terminology success rate metric. Ref: https://colab.research.google.com/drive/1b_M4tHnJxYT1JHZakjq44YJJIVsOP38h?usp=sharing 
# FAQ Section 4 (https://www2.statmt.org/wmt25/terminology.html) 


from fuzzywuzzy import fuzz
import stanza
import os
import json
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())


stanza_en = stanza.Pipeline('en', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_cs = stanza.Pipeline('cs', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_de = stanza.Pipeline('de', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_es = stanza.Pipeline('es', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_ru = stanza.Pipeline('ru', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_zh = stanza.Pipeline('zh', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_ja = stanza.Pipeline('ja', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_is = stanza.Pipeline('is', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
# stanza_km = stanza.Pipeline('km', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources") # not supported in stanza
# stanza_ha = stanza.Pipeline('ha', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources") # not supported in stanza
stanza_uk = stanza.Pipeline('uk', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_ta = stanza.Pipeline('ta', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_pl = stanza.Pipeline('pl', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_fr = stanza.Pipeline('fr', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_he = stanza.Pipeline('he', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
stanza_hr = stanza.Pipeline('hr', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources", download_method=None)
# stanza_ps = stanza.Pipeline('ps', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False, dir="./stanza_resources") # not supported in stanza


lang_dict = {'en': stanza_en, 'cs': stanza_cs, 'de': stanza_de, 'es': stanza_es, 'ru': stanza_ru, "zh": stanza_zh, "ja": stanza_ja, "is": stanza_is, "uk": stanza_uk, "ta": stanza_ta, "pl": stanza_pl, "fr": stanza_fr, "he": stanza_he, "hr": stanza_hr}


def stanza_lemmatize(sentence, tokenizer):
    '''
    lemmatizes one sentence.
    :param sentence: str, system output
    :param tokenizer: stanza.Pipeline object
    :returns lemmatized_sent: str, lemmatized sentence
    '''
    doc = tokenizer(sentence)
    # lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma is None:
                lemmas.append(word.text)
            else:
                lemmas.append(word.lemma)
    # print(f"Lemmatized from {sentence} to {lemmas}")
    lemmatized_sent = ' '.join(lemmas)
    return lemmatized_sent


def find_match(sentence, kw, mode='regex', threshold=90):
    '''
    matches a term within a sentence
    :param sentence: str
    :param kw: str, single term (may be MWE)
    :param mode: {'regex', 'fuzzy'} - whether we use regular expressions or FuzzyWuzzy to find a term
    :param threshold: [0-99], similarity threshold for FuzzyWuzzy to count it as a match (100 is complete match)
    :returns match: int, 1 if there is a match, 0 otherwise
    '''
    # regex match
    if mode == 'regex':
        #print(f'searching {kw} in {sentence}')
        try:
            regex_match = re.search(kw, sentence)
        #    print(regex_match)
        except:
            regex_match = re.search(re.escape(kw), re.escape(sentence))

        #print(regex_match)
        if regex_match is not None:
            match = 1
        else:
            match = 0

    # fuzzy match
    elif mode == 'fuzzy':
        fuzzy_match = fuzz.partial_ratio(kw, sentence)
        fuzzy_type = 'fuzzy'+str(threshold)
        if len(sentence) > 200:
            fuzzy_match = fuzzy_match * 2
        if fuzzy_match >= threshold:
            match = 1
        else:
            match = 0

    return match


def process_keywords_for_sentence(sentence, kw_list, lang, lemmatize=False, tokenizer=None):
    '''
    processing of a single sentence given its terminology.
    :param sentence: str, mt output
    :param kw_list: str (json), dictionary of keywords for a given sentence
    :param lang: str, language code
    :param lemmatize: bool, whether we apply the comparison of lemmatized words (important for morphologically-rich languages)
    :param tokenizer: stanza.Pipeline object, if lemmatize is True
    :returns kw_count: int, number of keywords
    :returns regex_count: int, number of regex matches
    :returns fuzzy90_count: int, number of fuzzy matches
    '''
    #print(sentence)
    #print(kw_list)
    if kw_list is np.nan:
        return 0, 0, 0
    if sentence is np.nan:
        sentence = ''
    # kw_list = re.sub(',+', ',', kw_list)
    # kw_list = json.loads(kw_list)
    kw_count = len(kw_list)
    regex_count, fuzzy90_count, fuzzy70_count = 0, 0, 0
    if lemmatize:
        sentence = stanza_lemmatize(sentence, tokenizer)
    for kw_pair in kw_list:
        kw = kw_pair[lang]
        if lemmatize:
            kw = stanza_lemmatize(kw, tokenizer)
        match_regex =  find_match(sentence, kw, mode='regex')
        match_fuzzy90 = find_match(sentence, kw, mode='fuzzy', threshold=90)
        #match_fuzzy70 = find_match(sentence, kw, mode='fuzzy', threshold=50)

        regex_count += match_regex
        fuzzy90_count += match_fuzzy90
        #fuzzy70_count += match_fuzzy70

    return kw_count, regex_count, fuzzy90_count#, fuzzy70_count


def compute_success_rate(df):
    '''
    counts percentages of term matches for each subtype of success rate metric (regex VS fuzzy match, surface forms VS lemmatized)
    :param df: pd.DataFrame, with columns 'kw_count', 'regex', 'fuzzy90', 'lem_regex', 'lem_fuzzy90'
    :returns regex_rate: float, percentage of regex matches
    :returns fuzzy90_rate: float, percentage of fuzzy matches
    :returns lem_regex_rate: float, percentage of lemmatized regex matches
    :returns lem_fuzzy90_rate: float, percentage of lemmatized fuzzy matches
    '''
    kw_total = df['kw_count'].sum()
    regex_total, fuzzy90_total = df['regex'].sum(), df['fuzzy90'].sum()
    lem_regex_total, lem_fuzzy90_total = df['lem_regex'].sum(), df['lem_fuzzy90'].sum()
    if kw_total == 0:
        return 0.0, 0.0, 0.0, 0.0
    regex_rate, fuzzy90_rate = regex_total/kw_total, fuzzy90_total/kw_total
    lem_regex_rate, lem_fuzzy90_rate = lem_regex_total/kw_total, lem_fuzzy90_total/kw_total
    return regex_rate, fuzzy90_rate, lem_regex_rate, lem_fuzzy90_rate

def _success_rate_cycle(list_of_files):
    '''
    applying metric to a list of files. at each step, a Pandas DataFrame is created
    :param list_of_files: list of filenames
    :returns: file success_rate_metrics.tsv with all statistics
    '''
    lang_dict = {'en': stanza_en, 'cs': stanza_cs, 'de': stanza_de, 'es': stanza_es, 'ru': stanza_ru}
    for file in list_of_files:
        try:
            print(f'processing file {file}')
            lang = file.split('.')[-2].split('-')[-1]
            print(lang)
            print(lang_dict[lang])
            df = pd.read_csv(f'data/sys_outputs/{file}', sep='\t')
            df['kw_count'], df['regex'], df['fuzzy90'] = zip(*df.progress_apply(lambda row: process_keywords_for_sentence(row.mt, row.terms, lang), axis=1))
            _, df['lem_regex'], df['lem_fuzzy90'] = zip(*df.progress_apply(lambda row: process_keywords_for_sentence(row.mt, row.terms, lang, lemmatize=True, tokenizer=lang_dict[lang]), axis=1))
            regex, fuzzy90, lem_regex, lem_fuzzy90 = compute_success_rate(df)

            line = '\t'.join([file, str(regex), str(fuzzy90), str(lem_regex), str(lem_fuzzy90)])
            with open('success_rate_metrics.tsv', 'a', encoding='utf-8') as f:
                f.write(line+'\n')
        except:
            print(f'PROBLEM WITH FILE {file}')
    return 'done'


def success_rate_cycle(all_translations):
    # format of all_translations:
    # {"de": [{"mt": ..., "terms": [{"de": "..."},...]},...], "es": [...], ...}
    output = {}
    
    for target_lang, translations in all_translations.items():
        df = pd.DataFrame(translations)
        print(f'processing {len(df)} translations for language {target_lang}')
        df['kw_count'], df['regex'], df['fuzzy90'] = zip(*df.progress_apply(lambda row: process_keywords_for_sentence(row.mt, row.terms, target_lang), axis=1))
        _, df['lem_regex'], df['lem_fuzzy90'] = zip(*df.progress_apply(lambda row: process_keywords_for_sentence(row.mt, row.terms, target_lang, lemmatize=True, tokenizer=lang_dict[target_lang]), axis=1))
        
        regex, fuzzy90, lem_regex, lem_fuzzy90 = compute_success_rate(df)
        output[target_lang] = {
            "regex": regex,
            "fuzzy90": fuzzy90,
            "lem_regex": lem_regex,
            "lem_fuzzy90": lem_fuzzy90
        }
    return output


if __name__ == "__main__":
    # example usage
    all_translations = {
        "zh": [
            {"mt": "这是一个测试句子。", "terms": [{"zh": "测试"}, {"zh": "句子"}]},
            {"mt": "另一个例子。", "terms": [{"zh": "例子"}]}
        ],
        # "jp": [
        #     {"mt": "これはテスト文です。", "terms": [{"jp": "テスト"}, {"jp": "文"}]},
        #     {"mt": "別の例。", "terms": [{"jp": "例"}]}
        # ]
    }
    
    term_success_rates = success_rate_cycle(all_translations)
    print(term_success_rates)
