# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import time
import copy
import torch
import random
import logging
import argparse
import numpy as np
from io import open
import shutil
from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import SoftTemplate
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
						  RobertaTokenizer, T5Config, T5ForConditionalGeneration)
from code_bleu import _code_bleu, compare
from bleu2 import _bleu
from my_lib_bugsinpy import read_prompt_examples, get_elapse_time, read_examples



generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5
}

def set_seed(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


def read_arguments():
	parser = argparse.ArgumentParser()

	# outdated parameters
	# parser.add_argument("--model_type", default=None, type=str, required=False,
	# 					help="Model type: e.g. roberta")
	# parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
	# 					help="Path to pre-trained model: e.g. roberta-base")

	# Required parameters
	parser.add_argument("--log_name", default=None, type=str, required=True)

	parser.add_argument("--model_name", default=None, type=str, required=True)

	parser.add_argument("--lang", default=None, type=str, required=True)

	parser.add_argument("--output_dir", default="./pr_model", type=str, required=False,
						help="The output directory where the model predictions and checkpoints will be written.")

	parser.add_argument("--data_dir", default="./data/bugsinpy", type=str,
						help="Path to the dir which contains processed data for some languages")

	# parser.add_argument("--lang", default=None, type=str, required=False,
	# 					help="language to summarize")
	parser.add_argument("--choice", type=int, required=True, 
						help="Choice of template.")
	parser.add_argument("--no_cuda", default=False, action='store_true',
					help="Avoid using CUDA when available")
	parser.add_argument('--visible_gpu', type=str, default="",
						help="use how many gpus")

	# parser.add_argument("--add_task_prefix", default=False, action='store_true',
	# 					help="Whether to add task prefix for T5 and codeT5")
	# parser.add_argument("--add_lang_ids", default=False, action='store_true',
	# 					help="Whether to add language prefix for T5 and codeT5")

	parser.add_argument("--num_train_epochs", default=20, type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--num_test_epochs", default=1, type=int,
						help="Total number of testing epochs to perform.")
	parser.add_argument("--train_batch_size", default=2, type=int,
						help="Batch size per GPU/CPU for training.")
	parser.add_argument("--eval_batch_size", default=2, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")

	# other arguments
	parser.add_argument("--load_model_path", default=None, type=str,
						help="Path to trained model: Should contain the .bin files")
	parser.add_argument("--config_name", default="", type=str,
						help="Pretrained config name or path if not the same as model_name")
	parser.add_argument("--tokenizer_name", default="", type=str,
						help="Pretrained tokenizer name or path if not the same as model_name")
	parser.add_argument("--max_source_length", default=512, type=int,
						help="The maximum total source sequence length after tokenization. Sequences longer "
							 "than this will be truncated, sequences shorter will be padded.")
	parser.add_argument("--max_target_length", default=512, type=int,
						help="The maximum total target sequence length after tokenization. Sequences longer "
							 "than this will be truncated, sequences shorter will be padded.")
	parser.add_argument("--warm_up_ratio", default=0.1, type=float)

	# controlling arguments
	parser.add_argument("--do_train", action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_test", action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--freeze", action='store_true', default=False,
						help="Whether to freeze the pre-trained model.")
	# parser.add_argument("--do_lower_case", action='store_true',
	# 					help="Set this flag if you are using an uncased model.")

	parser.add_argument("--learning_rate", default=5e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--beam_size", default=10, type=int,
						help="beam size for beam search")
	parser.add_argument("--weight_decay", default=0.0, type=float,
						help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	# parser.add_argument("--max_grad_norm", default=1.0, type=float,
	# 					help="Max gradient norm.")
	# parser.add_argument("--max_steps", default=-1, type=int,
	# 					help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	# parser.add_argument("--eval_steps", default=-1, type=int,
	# 					help="")
	# parser.add_argument("--train_steps", default=-1, type=int,
	# 					help="")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training: local_rank")
	parser.add_argument('--seed', type=int, default=42,
						help="random seed for initialization")
	parser.add_argument('--early_stop_threshold', type=int, default=10)

	# print arguments
	args = parser.parse_args()

	return args


def main(args):
	set_seed(args.seed)
	
	# data path
	train_filename = args.data_dir + "/" + "train.jsonl"	# train
	dev_filename = args.data_dir + "/" +  "validation.jsonl"	# valid
	test_filename = args.data_dir + "/" +  "test.jsonl"	# test
	# Setup CUDA, GPU & distributed training
	os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	# Initializes the distributed backend which will take care of synchronizing nodes/GPUs
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1

	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
				   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

	args.device = device

	# make dir if output_dir not exist
	if os.path.exists(args.output_dir) is False:
		os.makedirs(args.output_dir)

	# *********************************************************************************************************

	# read model --------------------------------------------------------------
	model_config = T5Config.from_pretrained(args.model_name)
	plm = T5ForConditionalGeneration.from_pretrained(args.model_name, config=model_config)
	tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
	WrapperClass = T5TokenizerWrapper

	# model_name = "roberta"
	# pretrainedmodel_path = "microsoft/codebert-base"
	# from openprompt.plms import load_plm
	# plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)
	# define template
	# promptTemplate = SoftTemplate(model=plm, tokenizer=tokenizer,
	# 									  text='{"placeholder":"text_a"} {"mask"} ', 
	# 									   num_tokens=50)
	promptTemplate = SoftTemplate(model=plm, tokenizer=tokenizer, initialize_from_vocab=True).from_file(f"./scripts/codet5p/soft_template_bugsinpy.txt", choice=args.choice)

	# get model
	model = PromptForGeneration(plm=plm, template=promptTemplate, freeze_plm=args.freeze, tokenizer=tokenizer,
									   plm_eval_mode=False)
	model.to(device)

	if args.load_model_path is not None:
		# load best checkpoint for best bleu
		output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
		if not os.path.exists(output_dir):
			raise Exception("Best bleu model does not exist!")

		model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))
		logger.info("reload model from {}".format(args.load_model_path))

	# parallel or distribute setting
	if args.local_rank != -1:
		# Distributed training
		try:
			# from apex.parallel import DistributedDataParallel as DDP
			from torch.nn.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError(
				"Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		model = DDP(model)
	elif args.n_gpu > 1:
		# multi-gpu training
		model = torch.nn.DataParallel(model)

	logger.info("Model created!!")

	# train part --------------------------------------------------------------
	if args.do_train:
		# Prepare training data loader
		train_examples = read_prompt_examples(train_filename)
		
		# take an example
		wrapped_example = promptTemplate.wrap_one_example(train_examples[0])
		logger.info(wrapped_example)

		train_data_loader = PromptDataLoader(
			dataset=train_examples,
			tokenizer=tokenizer,
			template=promptTemplate,
			tokenizer_wrapper_class=WrapperClass,
			max_seq_length=args.max_source_length,
			decoder_max_length=args.max_target_length,
			shuffle=True,
			teacher_forcing=True,
			predict_eos_token=True,
			batch_size=args.train_batch_size
		)

		# Prepare optimizer and schedule (linear warmup and decay)
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			 'weight_decay': args.weight_decay},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		t_total = (len(train_data_loader) // args.gradient_accumulation_steps) * args.num_train_epochs
		optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
		scheduler = get_linear_schedule_with_warmup(optimizer,
													num_warmup_steps=int(t_total * args.warm_up_ratio),
													num_training_steps=t_total)

		# Start training
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num epoch = %d", args.num_train_epochs)

		# used to save tokenized development data
		nb_tr_examples, nb_tr_steps, global_step, best_bleu, best_loss = 0, 0, 0, 0, 1e6
		early_stop_threshold = args.early_stop_threshold

		eval_dataloader = None
		dev_dataloader = None

		early_stop_count = 0
		for epoch in range(args.num_train_epochs):

			model.train()
			tr_loss = 0.0
			train_loss = 0.0

			# progress bar
			bar = tqdm(train_data_loader, total=len(train_data_loader))

			for batch in bar:
				batch = batch.to(device)

				loss = model(batch)

				if args.n_gpu > 1:
					loss = loss.mean()  # mean() to average on multi-gpu.
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps

				tr_loss += loss.item()
				train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
				bar.set_description("epoch {} loss {}".format(epoch, train_loss))

				nb_tr_steps += 1
				loss.backward()

				if nb_tr_steps % args.gradient_accumulation_steps == 0:
					# Update parameters
					optimizer.step()
					optimizer.zero_grad()
					scheduler.step()
					global_step += 1

			# to help early stop
			this_epoch_best = False

			if args.do_eval:
				# Eval model with dev dataset
				nb_tr_examples, nb_tr_steps = 0, 0

				if eval_dataloader is None:
					# Prepare training data loader
					eval_examples = read_prompt_examples(dev_filename)

					eval_dataloader = PromptDataLoader(
						dataset=eval_examples,
						tokenizer=tokenizer,
						template=promptTemplate,
						tokenizer_wrapper_class=WrapperClass,
						max_seq_length=args.max_source_length,
						decoder_max_length=args.max_target_length,
						shuffle=False,
						teacher_forcing=False,
						predict_eos_token=True,
						batch_size=args.eval_batch_size
					)
				else:
					pass

				logger.info("\n***** Running evaluation *****")
				logger.info("  Num examples = %d", len(eval_dataloader) * args.eval_batch_size)
				logger.info("  Batch size = %d", args.eval_batch_size)

				# Start Evaluating model
				model.eval()
				eval_loss = 0

				for batch in eval_dataloader:
					batch = batch.to(device)

					with torch.no_grad():
						loss = model(batch)

					eval_loss += loss.sum().item()

				# print loss of dev dataset
				result = {'epoch': epoch,
						  'eval_ppl': round(np.exp(eval_loss), 5),
						  'global_step': global_step + 1,
						  'train_loss': round(train_loss, 5)}

				for key in sorted(result.keys()):
					logger.info("  %s = %s", key, str(result[key]))
				logger.info("  " + "*" * 20)

				# save last checkpoint
				# last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
				# if not os.path.exists(last_output_dir):
				# 	os.makedirs(last_output_dir)

				# # Only save the model it-self
				# model_to_save = model.module if hasattr(model, 'module') else model

				# output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
				#torch.save(model_to_save.state_dict(), output_model_file)

				logger.info("Previous best ppl:%s", round(np.exp(best_loss), 5))

				# save best checkpoint
				if eval_loss < best_loss:
					this_epoch_best = True

					logger.info("Achieve Best ppl:%s", round(np.exp(eval_loss), 5))
					logger.info("  " + "*" * 20)
					best_loss = eval_loss
					# Save best checkpoint for best ppl
					# output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
					# if not os.path.exists(output_dir):
					# 	os.makedirs(output_dir)
					# output_model_file = os.path.join(output_dir, "pytorch_model.bin")
					#torch.save(model_to_save.state_dict(), output_model_file)

				# Calculate bleu
				this_bleu, dev_dataloader, _, _, _, _, _= calculate_bleu(dev_filename, args, tokenizer, device, model, promptTemplate, WrapperClass, is_test=False, dev_dataloader=dev_dataloader, best_bleu=best_bleu, eval_examples=eval_examples)

				if this_bleu > best_bleu:
					this_epoch_best = True

					logger.info(" Achieve Best bleu:%s", this_bleu)
					logger.info("  " + "*" * 20)
					best_bleu = this_bleu
					# Save best checkpoint for best bleu
					output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
					output_model_file = os.path.join(output_dir, "pytorch_model.bin")
					torch.save(model_to_save.state_dict(), output_model_file)

			# whether to stop
			if this_epoch_best:
				early_stop_count = 0
			else:
				early_stop_count += 1
				if early_stop_count == early_stop_threshold:
					logger.info("early stopping!!!")
					break

	# use dev file and test file ( if exist) to calculate bleu
	if args.do_test:
		# read model
		output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
		if not os.path.exists(output_dir):
			raise Exception("Best bleu model does not exist!")

		model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))
		logger.info("reload model from {}".format(output_dir))
		model.eval()

		this_bleu, eval_dataloader, ttl, exact_matched, syntax_matched, cleaned_matched, scores = calculate_bleu(test_filename, args, tokenizer, device, model, promptTemplate, WrapperClass, is_test=True)
		logger.info("  %s = %s " % ("Total", str(ttl)))
		logger.info("  %s = %s " % ("Exact Fixed", str(len(exact_matched))))
		logger.info(exact_matched)
		logger.info("  %s = %s " % ("Syntax Fixed", str(len(syntax_matched))))
		logger.info(syntax_matched)
		logger.info("  %s = %s " % ("Cleaned Fixed", str(len(cleaned_matched))))
		logger.info(cleaned_matched)	
		logger.info("  %s = %s " % ("codebleu", str(this_bleu)))
		logger.info(scores)
		output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
		# shutil.rmtree(output_dir)



def calculate_bleu(file_name, args, tokenizer, device, model, promptTemplate, WrapperClass, output_file_name=None, is_test=False, dev_dataloader=None,
				   best_bleu=None, eval_examples=None):
	logger.info("BLEU file: {}".format(file_name))

	# whether append posbugsinpy to result file
	if output_file_name is not None:
		output_file_name = "_" + output_file_name
	else:
		output_file_name = ""

	if is_test:
		file_prefix = "test"
	else:
		file_prefix = "dev"

	# if dev dataset has been saved
	# if (not is_test) and (dev_dataloader is not None):
	# if dev_dataloader:
	# 	eval_dataloader = dev_dataloader
	# else:
	# read texts
	eval_examples = read_prompt_examples(file_name)

	# only use a part for dev
	if not is_test:
		eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))

	eval_dataloader = PromptDataLoader(
		dataset=eval_examples,
		tokenizer=tokenizer,
		template=promptTemplate,
		tokenizer_wrapper_class=WrapperClass,
		max_seq_length=args.max_source_length,
		decoder_max_length=args.max_target_length,
		shuffle=False,
		teacher_forcing=False,
		predict_eos_token=False,
		batch_size=args.eval_batch_size
	)

	model.eval()

	# generate texts by source
	generated_texts = []
	groundtruth_sentence = []
	sources = []
	guids = []
	ev_ex = read_examples(file_name, args)
	for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
		batch = batch.to(device)
		with torch.no_grad():
			_, output_sentence = model.generate(batch,**generation_arguments)

			generated_texts.extend(output_sentence)
			groundtruth_sentence.extend(batch['tgt_text'])
			guids.extend(batch['guid'])

	# write to file

	with open(os.path.join(args.output_dir, file_prefix + ".output"), 'w') as f, open(
			os.path.join(args.output_dir, file_prefix + ".gold"), 'w') as f1,open(
			os.path.join(args.output_dir, file_prefix + ".source"), 'w') as f2:

		for ref, gold, source in zip(generated_texts, groundtruth_sentence, ev_ex):
			# ref = ref.replace('\t', ' ')

			f.write(ref.replace('\n',' ').replace('\r', ' ') + '\n')
			
			f1.write(gold.replace('\n',' ') + '\n')

			f2.write(source.source.replace('\n',' ') + '\n')
			
	# compute bleu
	this_bleu, exact_matched, cleaned_matched, syntax_matched, scores = _code_bleu(os.path.join(args.output_dir, file_prefix + ".gold"), os.path.join(args.output_dir, file_prefix + ".output"),os.path.join(args.output_dir, file_prefix + ".source"), args.lang)
	this_bleu = round(this_bleu,2)
	# normal_bleu = round(_bleu(os.path.join(args.output_dir, file_prefix + ".gold"), os.path.join(args.output_dir, file_prefix + ".output")),2)
	ttl = len(generated_texts)
	if is_test:
		logger.info("  %s = %s " % ("codebleu", str(this_bleu)))
		# logger.info("  %s = %s " % ("bleu-4", str(normal_bleu)))
		logger.info("  %s = %s " % ("Total", str(ttl)))
		logger.info("  %s = %s " % ("Exact Fixed", str(len(exact_matched))))
		logger.info(exact_matched)
		logger.info("  %s = %s " % ("Syntax Fixed", str(len(syntax_matched))))
		logger.info(syntax_matched)
		logger.info("  %s = %s " % ("Cleaned Fixed", str(len(cleaned_matched))))
		logger.info(cleaned_matched)
	
	else:
		logger.info("  %s = %s \t Previous best codebleu %s" % ("codebleu-4", str(this_bleu), str(best_bleu)))

	logger.info("  " + "*" * 20)

	return this_bleu, eval_dataloader, ttl, exact_matched, syntax_matched, cleaned_matched, scores


if __name__ == "__main__":
	my_args = read_arguments()

	# begin time
	begin_time = time.time()

	# logger for record
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)

	# write to file
	handler = logging.FileHandler(my_args.log_name)
	handler.setLevel(logging.INFO)
	logger.addHandler(handler)

	# write to console
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	logger.addHandler(console)

	# print args
	logger.info(my_args)

	main(my_args)

	logger.info("Finish training and take %s", get_elapse_time(begin_time))
