from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1") 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

import pdb; pdb.set_trace()