import json
import os
import subprocess

import pandas as pd


class BigBenchHard:

    def __init__(self, task_name: str, root: str = None, split: str = "train", *args, **kwargs):
        """
        Tasks from BIG-Bench Hard
        <https://github.com/suzgunmirac/BIG-Bench-Hard>

        The train, val, test splits were constructed from  50/100/100 examples.

        :param root (string): Root directory
        :param split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"`` and ``"test"``.
        """
        self.root = root
        self.split = split
        self.task_name = task_name
        self._check_or_download_dataset()
        assert split in ["train", "val", "test"]
        data_path = os.path.join(self.root, self.task_name, f"{split}.csv")
        self.data = pd.read_csv(data_path, index_col=0).to_dict(orient='records')

    def get_data(self):
        return self.data

    @staticmethod
    def get_task_description():
        return ("You will answer a reasoning question. Think step by step. The last line of your "
                "response should be of the following format: <Answer> VALUE </Answer> where VALUE is a "
                "numerical value.")

    def _check_or_download_dataset(self):
        data_path = os.path.join(self.root, self.task_name, f"{self.split}.csv")
        if os.path.exists(data_path):
            return

        os.makedirs(os.path.join(self.root, self.task_name), exist_ok=True)
        # Download the dataset
        # Download from https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/[task_name].json
        # and save it to self.root
        subprocess.call(
            [
                "wget",
                f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{self.task_name}.json",
                "-O",
                os.path.join(self.root, f"{self.task_name}.json")
            ]
        )
        # Separate to train, val, test
        data = json.load(open(os.path.join(self.root, f"{self.task_name}.json")))
        examples = data["examples"]
        train_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[:50]]
        val_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[50:150]]
        test_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[150:]]
        train_path = os.path.join(self.root, self.task_name, "train.csv")
        with open(train_path, "w") as f:
            pd.DataFrame(train_examples).to_csv(f)
        val_path = os.path.join(self.root, self.task_name, "val.csv")
        with open(val_path, "w") as f:
            pd.DataFrame(val_examples).to_csv(f)
        test_path = os.path.join(self.root, self.task_name, "test.csv")
        with open(test_path, "w") as f:
            pd.DataFrame(test_examples).to_csv(f)
