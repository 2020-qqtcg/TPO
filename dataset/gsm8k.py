import random

from datasets import load_dataset

TASK_PROMPT_GSM8K_COMPLEX = """You are an expert in prompt engineering and task analysis. Your goal is to optimize a system prompt 
so that it works effectively for a **specific type of task**, not just a specific instance.

Try various methods that you think might be helpful to build a better system prompt. 
Show your reason about why you optimize system prompt in this way.

## Task Description 
Task is about high quality linguistically diverse grade school math word problems. 
Task is about question answering on basic mathematical problems that require multi-step reasoning.

- These problems take between 2 and 8 steps to solve.
- Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.
- A bright middle school student should be able to solve every problem: from the paper, "Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable."
- Solutions are provided in natural language, as opposed to pure math expressions. From the paper: "We believe this is the most generally useful data format, and we expect it to shed light on the properties of large language models’ internal monologues""

## Task Example
Here is a example of task belongs to sepcific type: 
<Task> {task} </Task> 
Do not include information that is strongly tied to the task example. It is merely an instance of this task type and 
should be used for reference only.

## Original System Prompt
Here is original system prompt for input task:
<Prompt> {system_prompt} </Prompt>

## Output Format
Output your final optimized system prompt in the following format:
<Optimized> Optimized and generalized system prompt here </Optimized>
"""

TASK_PROMPT_GSM8K = """You are an expert in prompt engineering and task analysis. Your goal is to optimize a system prompt 
so that it works effectively for a **specific type of task**, not just a specific instance.

Try various methods that you think might be helpful to build a better system prompt. 

## Task Description 
Task is about high quality linguistically diverse grade school math word problems. 
Task is about question answering on basic mathematical problems that require multi-step reasoning.

- These problems take between 2 and 8 steps to solve.
- Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.
- A bright middle school student should be able to solve every problem: from the paper, "Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable."
- Solutions are provided in natural language, as opposed to pure math expressions. From the paper: "We believe this is the most generally useful data format, and we expect it to shed light on the properties of large language models’ internal monologues""

## Task Example
Here is a example of task belongs to sepcific type: 
<Task> {task} </Task> 
Do not include information that is strongly tied to the task example. It is merely an instance of this task type and 
should be used for reference only.

## Output Format
Only output your final optimized system prompt.
"""

GSM8K_REASON = """You will answer a mathemetical reasoning question. Think step by step.

{task}
"""

GET_CHOICE_GSM8K = """
{task}

Here is the reasoning to the task above. Please determine the final number answer based on this reasoning.
Reasoning: {answer}

Output your final number answer format like:
<Answer> Your number answer </Answer>
"""


class GSM8K:
    def __init__(self):
        gsm8k = load_dataset("openai/gsm8k", "main")
        self.train_data = gsm8k["train"]
        self.test_data = gsm8k["test"]

    def random_select(self, rate: float) -> list:
        """
        Randomly select a subset of GSM8K training data.
        :param rate: Rate of selecting a subset of GSM8K (e.g., 0.1 for 10%)
        :return: List of randomly selected examples
        """
        if not (0 < rate <= 1):
            raise ValueError("Rate must be between 0 and 1.")

        total = len(self.train_data)
        num_samples = int(total * rate)
        selected_indices = random.sample(range(total), num_samples)
        selected_data = [self.train_data[i] for i in selected_indices]
        return selected_data


class GSM8KDSPY:
    def __init__(self, split: str = "train", mode: str = "prompt"):
        """DSPy splits for the GSM8K dataset."""
        import tqdm
        import random
        from datasets import load_dataset

        dataset = load_dataset("gsm8k", 'main')
        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'

            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))

            if mode == "prompt":
                official_train.append(dict(x=question, gold_reasoning=gold_reasoning, y=answer))
            elif mode == "reasoning":
                question = GSM8K_REASON.format(task=question)
                official_train.append(dict(task=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']
            answer = example['answer'].strip().split()
            assert answer[-2] == '####'

            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))
            if mode == "prompt":
                official_test.append(dict(x=question, gold_reasoning=gold_reasoning, y=answer))
            elif mode == "reasoning":
                question = GSM8K_REASON.format(task=question)
                official_test.append(dict(task=question, gold_reasoning=gold_reasoning, answer=answer))

        rng = random.Random(0)
        rng.shuffle(official_train)
        rng = random.Random(0)
        rng.shuffle(official_test)
        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]
        if split == "train":
            self.data = trainset
        elif split == "val":
            self.data = devset
        elif split == "test":
            self.data = testset

    def get_data(self):
        return self.data

    @staticmethod
    def get_task_description():
        return "You will answer a mathemetical reasoning question. Think step by step."

    @staticmethod
    def get_task_output_format():
        return ("The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is the "
                "numerical value.")

    @staticmethod
    def get_task_defination():
        return TASK_PROMPT_GSM8K
