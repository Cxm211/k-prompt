import json
import time

from openprompt.data_utils import InputExample
import difflib

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename, args):
    """Read examples from filename."""
    # if args.add_task_prefix:
    #     task_prefix = f"Translate "
    # else:
    #     task_prefix = ""

    # if args.add_lang_ids:
    #     language_prefix = "<en> "
    # else:
    #     language_prefix = ""

    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            buggy = js['problem']
            fixed = js['fixed']
            
            # code = js['code'].replace('\n', ' ').strip().replace('\t', ' ')
            # nl = js['docstring'].replace('\n', ' ').replace('\t', ' ').strip()
            examples.append(
                Example(
                    idx=idx,
                    source= buggy, #+ ' the fixed version ',
                    target= fixed,
                )
            )

    return examples


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # collect texts
    codes = []
    target_nl = []
    for example_id, example in enumerate(examples):
        codes.append(example.source)

        if stage == "test":
            target_nl.append("None")
        else:
            target_nl.append(example.target)

    # begin tokenizing
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    return {'source_ids':encoded_codes['input_ids'], 'target_ids':encoded_targets['input_ids'],
            'source_mask':encoded_codes['attention_mask'], 'target_mask':encoded_targets['attention_mask']}


def read_prompt_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            buggy = js['problem'].replace('\n', ' ').replace('\r', ' ').strip()
            fixed = js['fixed'].replace('\n', ' ').replace('\r', ' ').strip()
            if "hint" in js:
                hint = js['hint']
            else:
                hint = ""
            testcases = js['testcases']
            hint_list = hint.split("\n")
            new = []
            try:
                a = hint_list.index("Input:")
                b = hint_list.index("Output:")
                new.append(hint_list[a])
                new.append(hint_list[a+1])
                new.append(hint_list[b])
                new.append(hint_list[b+1])
                hint = "\n".join(new)
            except:
                hint = ''
            if testcases!=[]:
                line = testcases[0].strip()
                data = json.loads(line)
                input_data = str(data[0])  
                output_data = str(data[1])
            else:
                input_data = ""
                output_data = ""
            # parsed_testcases.append(input_output)
            # # =code = js['code'].replace('\n', ' ').strip()
            # # nl = js['docstring'].replace('\n', ' ').strip()
            # if parsed_testcases == []:
            #     input_string = ''
            #     output_string = ''
            # else:
            #     input = parsed_testcases[0][0]
            #     output = parsed_testcases[0][1]
            #     if type(input) == type([]):
            #         list_items_str = [str(item) for item in input]
            #         input_string = ', '.join(list_items_str)
            #     else:
            #         input_string = str(input)
            #     if type(output) == type([]):
            #         list_items_str = [str(item) for item in output]
            #         output_string = ', '.join(list_items_str)
            #     else:
            #         output_string = str(output)
            if len(input_data) >=500:
                 input_data = ""
            if len(output_data) >=500:
                 output_data = ""
            examples.append(
                InputExample(
                    guid=idx,
                    text_a= buggy,
                    tgt_text= fixed,
                    text_b= hint,
                    meta={
                        "hint":hint,
                        "input": input_data[1:-1],
                        "output": output_data
                    }
                )
            )

    return examples

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

        diff = 0
        list_diff = []

        for line_num, (line1, line2, json_line) in enumerate(zip(file1_lines, file2_lines, jsonl_lines), 1):

            json_data = json.loads(json_line)
            problem = json_data['problem'].replace('\n','').replace("\t","").replace("\r","").replace(" ","").replace("{","").replace("}","")
            line2 = line2.replace('\n','').replace("\t","").replace("\r","").replace(" ","").replace("{","").replace("}","")

            matcher = difflib.SequenceMatcher(None, problem, line2)
            differences = [tag for tag, _, _, _, _ in matcher.get_opcodes() if tag != 'equal']

            if not differences:
                
                diff += 1 
            else:
                list_diff.append(line_num)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != 'equal':
                        pass
                        
                #         print(f"Line {line_num}:", end='\n')
                #         print(f"Problem: {problem[i1-20:i2+20]}", end='\n')
                #         print(f"File2: {line2[j1-20:j2+20]}", end='\n')
                # print("\n")

        print("Ttl: " + str(len(jsonl_lines)), file=output_file)
        print("Fixed: " + str(equal), file=output_file)
        print(list_eq, file=output_file)
        print("Unchange: " + str(diff), file=output_file)
        print(list_diff, file=output_file)
    return len(jsonl_lines), equal, list_eq, diff,list_diff