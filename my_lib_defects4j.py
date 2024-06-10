import json
import time
import difflib
from openprompt.data_utils import InputExample

REF = {
    "mdAdd": "Method definition addition",
    "mdRem": "Method definition removal",
    "mdRen": "Method definition renaming",
    "mdParAdd": "Parameter addition in method definition",
    "mdParRem": "Parameter removal from method definition",
    "mdRetTyChange": "Method return type modification",
    "mdParTyChange": "Parameter type modification in method definition",
    "mdModChange": "Method modifier change",
    "mdOverride": "Method overriding addition or removal",
    "mcAdd": "Method call addition",
    "mcRem": "Method call removal",
    "mcRepl": "Method call replacement",
    "mcParSwap": "Method call parameter value swapping",
    "mcParAdd": "Method call parameter addition",
    "mcParRem": "Method call parameter removal",
    "mcParValChange": "Method call parameter value modification",
    "mcMove": "Method call moving",
    "objInstAdd": "Object instantiation addition",
    "objInstRem": "Object instantiation removal",
    "objInstMod": "Object instantiation modification",
    "varAdd": "Variable addition",
    "varRem": "Variable removal",
    "varReplVar": "Variable replacement by another variable",
    "exTryCatchAdd": "try-catch addition",
    "exTryCatchRem": "try-catch removal",
    "exThrowsAdd": "throw addition",
    "exThrowsRem": "throw removal",
    "condExpRed": "Conditional expression reduction",
    "condExpExpand": "Conditional expression expansion",
    "condExpMod": "Conditional expression modification",
    "condBranIfAdd": "Conditional (if) branch addition",
    "condBranIfElseAdd": "Conditional (if-else) branches addition",
    "condBranElseAdd": "Conditional (else) branch addition",
    "condBranCaseAdd": "Conditional (case in switch) branch addition",
    "condBranRem": "Conditional (if or else) branch removal",
    "assignAdd": "Assignment addition",
    "assignRem": "Assignment removal",
    "assignExpChange": "Assignment expression modification",
    "loopAdd": "Loop addition",
    "loopRem": "Loop removal",
    "loopCondChange": "Loop conditional expression modification",
    "loopInitChange": "Loop initialization field modification",
    "varTyChange": "Variable type change",
    "varModChange": "Variable modifier change",
    "varReplMc": "Variable replacement by method call",
    "tyAdd": "Type addition",
    "tyImpInterf": "Type implemented interface modification",
    "retExpChange": "Return expression modification",
    "retBranchAdd": "Return statement addition",
    "retRem": "Return statement removal",
    "wrapsIf": "Wraps-with if statement",
    "wrapsIfElse": "Wraps-with if-else statement",
    "wrapsElse": "Wraps-with else statement",
    "wrapsTryCatch": "Wraps-with try-catch block",
    "wrapsMethod": "Wraps-with method call",
    "wrapsLoop": "Wraps-with loop",
    "unwrapIfElse": "Unwraps-from if-else statement",
    "unwrapMethod": "Unwraps-from method call",
    "unwrapTryCatch": "Unwraps-from try-catch block",
    "condBlockExcAdd": "Conditional block addition with exception throwing",
    "condBlockRetAdd": "Conditional block addition with return statement",
    "condBlockOthersAdd": "Conditional block addition",
    "condBlockRem": "Conditional block removal",
    "missNullCheckP": "Missing null check addition",
    "missNullCheckN": "Missing non-null check addition",
    "expLogicExpand": "Logic expression expansion",
    "expLogicReduce": "Logic expression reduction",
    "expLogicMod": "Logic expression modification",
    "expArithMod": "Arithmetic expression modification",
    "codeMove": "Code Moving",
    "wrongVarRef": "Wrong Variable Reference",
    "wrongMethodRef": "Wrong Method Reference",
    "singleLine": "Single Line",
    "notClassified": "Not classified",
    "copyPaste": "Copy/Paste",
    "constChange": "Constant Change",
    "rtAcs": "Patched by ACS",
    "rtCardumen": "Patched by Cardumen",
    "rtDeepRepair": "Patched by DeepRepair",
    "rtDynaMoth": "Patched by DynaMoth",
    "rtElixir": "Patched by Elixir",
    "rtGPFL": "Patched by GPFL",
    "rtHDRepair": "Patched by HDRepair",
    "rtGenProg": "Patched by jGenProg",
    "rtKali": "Patched by jKali",
    "rtNopol": "Patched by Nopol",
    "rtssFix": "Patched by ssFix",
}

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
            buggy = js['problem'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            fixed = js['fixed'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            
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
            buggy = js['problem'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            fixed = js['fixed'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            observations = js['observations']
            repairAction = ''
            for action in js['repairActions']:
                if action in REF:
                    repairAction += REF[action]
                    repairAction += ", "
            repairPattern = ''
            for action in js['repairPatterns']:
                if action in REF:
                    repairPattern += REF[action]
                    repairPattern += ","
            # if len(observations) > 80:
            #      observations = ""
            examples.append(
                InputExample(
                    guid=idx,
                    text_a= buggy,
                    tgt_text= fixed,
                    meta = {
                        "observations": observations,
                        "repairAction": repairAction,
                        "repairPattern": repairPattern
                    }
                )
            )
    print("Total length:" + str(len(examples)))

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