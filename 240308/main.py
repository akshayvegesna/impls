import re
import os
import pandas as pd
import copy 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task:
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        path = file
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

PROMPT = """ 
You are an agent tasked with solving the game of 24.

The game of 24 is a mathematical card game in which the goal is to find a way to manipulate four numbers so that the result is 24. Each of the four numbers must be used exactly once, and you can use the basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/). The game is often played with cards, each bearing a number, and the challenge is to figure out how to combine these numbers to make 24.

===
EXAMPLE: 
PROBLEM: 4 7 8 8 
SOLUTION: (7 - (8 / 8)) * 4
===
PROBLEM: $PROBLEM
SOLUTION:"""

def gen_samples(llm, sampling_params, tokenizer, task):

    prompts = []


    problem = task.get_input(0)
    prompt = copy.deepcopy(PROMPT).replace("$PROBLEM", problem)
    print('Prompt:', prompt)
    prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    outputs_filtered = [[(output.text, output.logprobs) for output in output_group.outputs] for output_group in outputs]

    import pdb; pdb.set_trace()


def main(): 
    # let's just run llama 7b to try to solve each one of these. 


    model_name = 'mistralai/Mistral-7B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tensor_parallel_size = 1
    max_tokens = 200
    temperature = 0.87
    top_p = 0.95
    n_samples = 5

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, n=n_samples, max_tokens=max_tokens, logprobs=1, stop=['---\n'] )
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    task = Game24Task()

    gen_samples(llm, sampling_params, tokenizer, task)



if __name__ == '__main__': 
    main()