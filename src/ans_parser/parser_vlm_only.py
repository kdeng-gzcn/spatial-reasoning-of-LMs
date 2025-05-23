import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

from .parser_template import ParserTemplate

# Subclass for `012` classification task (Uncertain, Left, Right, )
class Metric012Baseline(ParserTemplate):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _extract_info(self, text):

        ans_match = re.search(r"<ans>.*?(\d+).*?</ans>", text, re.IGNORECASE)
        ans = int(ans_match.group(1) if ans_match else None)

        rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|$|\s*<ques>)", text, re.DOTALL)
        rsn = rsn_match.group(1) if rsn_match else "None"

        ques_match = re.search(r"<ques>\s*(.*?)(?:\s*</ques>|$)", text, re.DOTALL)
        ques = ques_match.group(1) if ques_match else "None"

        return ans, rsn, ques

    def _process_conclusion(self):

        self.ans, self.rsn, self.ques = self._extract_info(self.conclusion)

        if self.ans is None:
            print("Answer Option Not Extracted, Random Selection")
            self.ans = 0

        self.result_one_round = {
            "scene": self.metadata["scene"],
            "seq": self.metadata["seq"],
            "pair": self.metadata["pair"],
            "dof": self.metadata["significance"],
            "label text": self.metadata["significance_text"],
            "label value": self.metadata["significance_value"],
            "pred option": self.ans,
            "pred text": self.option_map[self.ans],
            "reason": self.rsn,
            "question": self.ques,
        }

        self.result_dict.append(self.result_one_round)

    def __call__(self, conclusion_from_LLM, metadata_dict, mapping):

        self.option_map = mapping

        self.conclusion = conclusion_from_LLM
        self.metadata = metadata_dict

        self._process_conclusion()
    
    def _evaluate(self):

        # Global
        results = Stat012(self.result_dict)._calculate_metrics()

        return results


class Stat012:

    def __init__(self, result_dict):
        
        self.data = pd.DataFrame(result_dict)

    def _calculate_metrics(self):

        stat = {}

        total_samples = len(self.data)
        count_zero = len(self.data[self.data["pred text"] == "unable to judge"])
        zero_percentage = count_zero / total_samples * 100

        filtered_data = self.data[self.data["pred text"] != "unable to judge"]

        y_true = filtered_data["label text"]
        y_pred = filtered_data["pred text"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="leftward")
        recall = recall_score(y_true, y_pred, pos_label="leftward")
        f1 = f1_score(y_true, y_pred, pos_label="leftward")

        # Confusion Matrix
        labels = ["leftward", "rightward"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        # Return the metrics as a dictionary
        scalar_metrics = {
            "length of dataset": [total_samples],
            "unable to answer percentage": [zero_percentage],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
        }

        metrics = {
            "scalar metrics": scalar_metrics,
            "confusion matrix": cm_df
        }

        stat[f"summary stat"] = metrics

        return stat
    

class Metric0123Baseline(ParserTemplate):
    def __init__(self):
        super().__init__()

    def _extract_info(self, text):
        rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|\s*<ans>|$)", text, re.DOTALL)
        rsn = rsn_match.group(1) if rsn_match else "None"

        ans_match = re.search(r"<ans>.*?(\d+).*?(?:</ans>|$)", text, re.IGNORECASE)
        ans = int(ans_match.group(1)) if ans_match else None

        ques = "None"
        return ans, rsn, ques

    def _process_conclusion(self):
        self.ans, self.rsn, self.ques = self._extract_info(self.conclusion)
        if self.ans is None or self.ans not in [0, 1, 2, 3]: # avoid NoneType error
            print("Answer Option Not Extracted")
            self.ans = next(key for key, value in self.option_map.items() if value == "unable to judge")

        # self.result_one_round = {
        #     "scene": self.metadata["scene"],
        #     "seq": self.metadata["seq"],
        #     "pair": self.metadata["pair"],
        #     "dof": self.metadata["significance"],
        #     "label text": self.metadata["significance_text"],
        #     "label value": self.metadata["significance_value"],
        #     "pred option": self.ans,
        #     "pred text": self.option_map[self.ans],
        #     "reason": self.rsn,
        #     "question": self.ques,
        # }

        # self.result_dict.append(self.result_one_round)

    def __call__(self, conclusion_from_LLM: str, metadata_dict: dict, mapping: dict) -> dict:
        self.option_map = mapping
        self.conclusion = conclusion_from_LLM
        self.metadata = metadata_dict

        self._process_conclusion()

        result = {
            "pred option": self.ans,
            "pred text": self.option_map[self.ans],
            "reason": self.rsn,
            "question": self.ques,
        }

        return result
