import random

from datasets import load_dataset

task_template = """
Answer the following multiple choice question. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
"""

GET_CHOICE = """
{task}

Here is the answer to the question above. Please determine the final corresponding option based on this answer. You must choose an option of A or B or C or D. 
Answer: {answer}

Output your final choice format like:
<Answer> A or B or C or D </Answer>
"""


class GPQA:

    def __init__(self, subset: str = "gpqa_diamond"):
        self.data = load_dataset("Idavidrein/gpqa", subset, split="train")

        self.filterd_data = []
        for row in self.data:
            choices = [
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3'],
                row['Correct Answer']
            ]
            choices = [choice.strip() for choice in choices]

            random.shuffle(choices)
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            correct_answer_idx = choices.index(row['Correct Answer'].strip())

            task = task_template.format(
                Question=choices_dict['Question'],
                A=choices_dict['A'],
                B=choices_dict['B'],
                C=choices_dict['C'],
                D=choices_dict['D'])

            answer = "A" if correct_answer_idx == 0 \
                else "B" if correct_answer_idx == 1 \
                else "C" if correct_answer_idx == 2 \
                else "D"

            self.filterd_data.append({
                "task": task,
                "answer": answer,
                "goledn_answer": row["Explanation"]
            })

        if subset == "gpqa_main":
            random.seed(42)
            random.shuffle(self.filterd_data)
            self.filterd_data = self.filterd_data[:200]
