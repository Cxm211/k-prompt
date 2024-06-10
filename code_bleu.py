from codebleu import calc_codebleu
import json 
import difflib
from CodeBLEU import syntax_match

def is_perfect_match(result):
    return result['syntax_match_score'] == 1.0 and result['dataflow_match_score'] == 1.0

def compute_code_bleu(refs, hyp, source, lang, alpha, beta, gamma, theta):
    # preprocess inputs

    references = [x.strip().replace(" . ",". ").replace(" , ",", ") for x in open(refs, 'r', encoding='utf-8').readlines()]
    hypothesis = [x.strip() for x in open(hyp, 'r', encoding='utf-8').readlines()]
    sources = [x.strip().replace(" . ",". ").replace(" , ",", ") for x in open(source, 'r', encoding='utf-8').readlines()]
    ref1 = [x.strip().replace('\n','').replace("\t","").replace("\r","").replace(" ","") for x in open(refs, 'r', encoding='utf-8').readlines()]
    hyp1 = [x.strip().replace('\n','').replace("\t","").replace("\r","").replace(" ","") for x in open(hyp, 'r', encoding='utf-8').readlines()]
    sour1 = [x.strip().replace('\n','').replace("\t","").replace("\r","").replace(" ","") for x in open(source, 'r', encoding='utf-8').readlines()]


    cleaned_ref = [x.replace('\n','').replace("\t","").replace("\r","").replace("{","").replace("}","").strip().replace(" ","") for x in open(refs, 'r', encoding='utf-8').readlines()]
    cleaned_hyp = [x.replace('\n','').replace("\t","").replace("\r","").replace("{","").replace("}","").strip().replace(" ","") for x in open(hyp, 'r', encoding='utf-8').readlines()]
    assert len(hypothesis) == len(references)
    syntax_matched = []
    exact_matched = []
    cleaned_matched = []
    scores = []
    for i in range(len(hypothesis)):
        result = calc_codebleu([hypothesis[i]], [references[i]], lang, weights=(alpha, beta, gamma, theta), tokenizer=None)
        source_re = calc_codebleu([hypothesis[i]], [sources[i]], lang, weights=(alpha, beta, gamma, theta), tokenizer=None)
        source_re1 = calc_codebleu([references[i]], [sources[i]], lang, weights=(alpha, beta, gamma, theta), tokenizer=None)
        cleaned_result = calc_codebleu([cleaned_hyp[i]], [cleaned_ref[i]], lang, weights=(alpha, beta, gamma, theta), tokenizer=None)
        if ref1[i] == hyp1[i]:
            exact_matched.append(i+1)
        if is_perfect_match(result) and source_re['ngram_match_score'] != 1 and sour1[i] != hyp1[i] and not is_perfect_match(source_re1):
            syntax_matched.append(i+1)
        if cleaned_hyp[i] == cleaned_ref[i]:
            cleaned_matched.append(i+1)
        if result['dataflow_match_score'] == 0:
            result['codebleu'] = alpha*result['ngram_match_score']\
                    + beta*result['weighted_ngram_match_score']\
                    + gamma*result['syntax_match_score']\
                    + theta*result['syntax_match_score']
        scores.append(result['codebleu'])
    for i in exact_matched:
        if i in syntax_matched:
            syntax_matched.remove(i)
        if i in cleaned_matched:
            cleaned_matched.remove(i)
    bleu = sum(scores) / len(scores)
    return bleu, exact_matched, cleaned_matched,syntax_matched, scores
    

def _code_bleu(ref, hyp, source, lang):
    bleu, exact_matched, cleaned_matched,syntax_matched, scores = compute_code_bleu(ref, hyp, source, lang, 0.2, 0.2, 0.3, 0.3)
    return round(100 * bleu,2), exact_matched, cleaned_matched, syntax_matched, scores

def compare(file1_path, file2_path, jsonl_path, output):

    # file1_path = 'defects4j/SC_finetune_codet5p_770m/test.gold'
    # file2_path = 'defects4j/SC_finetune_codet5p_770m/test.output'
    # jsonl_path = 'data/defects4j_SC/test.jsonl'
    with open(output, 'w') as output_file:
        with open(jsonl_path, 'r') as jsonl_file:
            jsonl_lines = jsonl_file.readlines()

        with open(file1_path, 'r') as file1:
            file1_lines = file1.readlines()

        with open(file2_path, 'r') as file2:
            file2_lines = file2.readlines()
        equal =0
        list_eq = []
        for line_num, (line1, line2, json_line) in enumerate(zip(file1_lines, file2_lines, jsonl_lines), 1):

            json_data = json.loads(json_line)
            problem = json_data['problem'].replace('\n','').replace("\t","").replace("\r","").replace(" ","").replace("{","").replace("}","")
            line1_clean = line1.replace('\n','').replace("\t","").replace("\r","").replace(" ","").replace("{","").replace("}","")
            line2_clean = line2.replace('\n','').replace("\t","").replace("\r","").replace(" ","").replace("{","").replace("}","")


            matcher1 = difflib.SequenceMatcher(None, line1_clean, line2_clean)
            if not any(tag != 'equal' for tag, _, _, _, _ in matcher1.get_opcodes()):
                equal += 1
                list_eq.append(line_num)
            else:
                for tag, i1, i2, j1, j2 in matcher1.get_opcodes():
                    if tag != 'equal':
                        print(f"Line {line_num} (File1 vs File2):", file=output_file)
                        print(f"Ground: {line1_clean[i1-20:i2+20]}", end='\n', file=output_file)
                        print(f"Output: {line2_clean[j1-20:j2+20]}", end='\n', file=output_file)


            matcher2 = difflib.SequenceMatcher(None, problem, line1_clean)
            if not any(tag != 'equal' for tag, _, _, _, _ in matcher2.get_opcodes()):
                pass
            else:
                for tag, i1, i2, j1, j2 in matcher2.get_opcodes():
                    if tag != 'equal':
                        print(f"Line {line_num} (Problem vs File1):", file=output_file)
                        print(f"Ground: {line1_clean[i1-20:i2+20]}", end='\n', file=output_file)
                        print(f"Problem: {problem[i1-20:i2+20]}", end='\n', file=output_file)

        unchange = 0
        list_unchange = []

        for line_num, (line1, line2, json_line) in enumerate(zip(file1_lines, file2_lines, jsonl_lines), 1):

            json_data = json.loads(json_line)

            import re
            normalized_text1 = re.sub(r'[\s{}]+', '', json_data['problem'])
            normalized_text2 = re.sub(r'[\s{}]+', '', line2)

            if normalized_text1 == normalized_text2:
                list_unchange.append(line_num)
                unchange += 1 
            # else:
            #     list_diff.append(line_num)
            #     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            #         if tag != 'equal':
            #             pass
                        
                #         print(f"Line {line_num}:", end='\n')
                #         print(f"Problem: {problem[i1-20:i2+20]}", end='\n')
                #         print(f"File2: {line2[j1-20:j2+20]}", end='\n')
                # print("\n")

        print("Ttl: " + str(len(jsonl_lines)), file=output_file)
        print("Fixed: " + str(equal), file=output_file)
        print(list_eq, file=output_file)
        print("Unchange: " + str(unchange), file=output_file)
        print(list_unchange, file=output_file)
    return len(jsonl_lines), equal, list_eq, unchange, list_unchange