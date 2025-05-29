import random

from datasets import load_dataset

task_template = """
Answer the following multiple choice question. Think step bt step before answering.
The last line of your response should be of the following format: <Answer> LETTER </Answer> where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
"""


class MMLU:

    def __init__(self, subset: str = "machine_learning", split: str = "test"):
        self.data = load_dataset("cais/mmlu", subset, split=split)

        self.filterd_data = []
        for row in self.data:
            question = row["question"]
            choices = row["choices"]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=question
            )
            question_prompt = task_template.format(**choices_dict)

            # Choices will be a. Choice 1 b. Choice 2 ... etc
            answer = chr(65 + row["answer"])

            self.filterd_data.append({
                "task": question_prompt,
                "answer": answer
            })


