import re
from typing import Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .parser_template import ParserTemplate

class Metric012(ParserTemplate):
    def __init__(self, **kwargs):
        super().__init__()
        self.result_dict = []

    def __call__(self, round_num, conclusion_from_LLM, metadata_dict):

        self.round_num = round_num
        self.conclusion = conclusion_from_LLM
        self.metadata = metadata_dict

        self._process_conclusion()

    def _extract_info(self, text):

        ans_match = re.search(r"<ans>.*?(\d+).*?</ans>", text)
        ans = int(ans_match.group(1)) if ans_match else None

        rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|$|\s*<ques>)", text, re.DOTALL)
        rsn = rsn_match.group(1) if rsn_match else "None"

        ques_match = re.search(r"<ques>\s*(.*?)(?:\s*</ques>|$)", text, re.DOTALL)
        ques = ques_match.group(1) if ques_match else "None"

        return ans, rsn, ques

    def _process_conclusion(self):

        self.ans, self.rsn, self.ques = self._extract_info(self.conclusion)

        if self.ans is None: # avoid

            print("Answer Option Not Extracted")

            self.ans = 0

        self.result_one_round = {
            "scene": self.metadata["scene"],
            "seq": self.metadata["seq"],
            "pair": self.metadata["pair"],
            "dof": self.metadata["significance"],
            "label text": self.metadata["significance_text"],
            "label value": self.metadata["significance_value"],
            "round idx": self.round_num,
            "pred option": self.ans,
            "pred text": self.option_map[self.ans],
            "reason": self.rsn,
            "question": self.ques,
        }

        self.result_dict.append(self.result_one_round)
    
    def _evaluate(self):

        stat = Stat012Conv(self.result_dict)

        results = stat._calculate_metrics()

        return results

# Class to calculate evaluation metrics for the `012` classification task
class Stat012Conv:

    def __init__(self, result_dict):

        self.data = pd.DataFrame(result_dict)

    def _calculate_metrics(self):

        max_round = self.data["round idx"].max()

        stat = {}

        summary = self.data.loc[self.data.groupby(['scene', 'seq', 'pair'])['round idx'].idxmax()]

        total_samples = len(summary)
        count_zero = len(summary[summary["pred option"] == 0])
        zero_percentage = count_zero / total_samples * 100

        filtered_data = summary[summary["pred option"] != 0]

        y_true = filtered_data["label text"]
        y_pred = filtered_data["pred text"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="leftward")
        recall = recall_score(y_true, y_pred, pos_label="leftward")
        f1 = f1_score(y_true, y_pred, pos_label="leftward")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=["leftward", "rightward"])

        # Return the metrics as a dictionary
        scalar_metrics = {
            # "info": ["phi (left, right), 1 round experiment"],
            "length of dataset": [total_samples],
            "not finished percentage": [zero_percentage],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
        }

        metrics = {
            "scalar metrics": scalar_metrics,
            "confusion matrix": cm
        }

        stat[f"summary stat"] = metrics

        for round in range(1, max_round+1):

            round_stat = self.data[self.data["round idx"] == round]

            total_samples = len(round_stat)
            count_zero = len(round_stat[round_stat["pred option"] == 0])
            zero_percentage = count_zero / total_samples * 100

            filtered_data = round_stat[round_stat["pred option"] != 0]

            y_true = filtered_data["label text"]
            y_pred = filtered_data["pred text"]

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label="leftward")
            recall = recall_score(y_true, y_pred, pos_label="leftward")
            f1 = f1_score(y_true, y_pred, pos_label="leftward")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=["leftward", "rightward"])

            # Return the metrics as a dictionary
            scalar_metrics = {
                # "info": ["phi (left, right), 1 round experiment"],
                "length of dataset": [total_samples],
                "not finished percentage": [zero_percentage],
                "accuracy": [accuracy],
                "precision": [precision],
                "recall": [recall],
                "f1_score": [f1],
            }

            metrics = {
                "scalar metrics": scalar_metrics,
                "confusion matrix": cm
            }

            stat[f"{round} round stat"] = metrics

        return stat
    
class Metric0123Conv(ParserTemplate):
    def __init__(self):
        super().__init__()

    def _extract_info(self, text: str) -> Tuple[int, str, str]:
        rsn_match = re.search(r"<rsn>\s*(.*?)(?:\s*</rsn>|\s*<ans>|$)", text, re.DOTALL)
        rsn = rsn_match.group(1) if rsn_match else "None"

        ques_match = re.search(r"<ques>\s*(.*?)(?:\s*</ques>|\s*<ans>|$)", text, re.DOTALL)
        ques = ques_match.group(1) if ques_match else "None"

        ans_match = re.search(r"<ans>.*?(\d+).*?(?:</ans>|$)", text)
        ans = int(ans_match.group(1)) if ans_match else None
        return rsn, ques, ans

    def _process_conclusion(self, text: str, mapping: dict) -> Tuple[int, str, str]:
        rsn, ques, ans = self._extract_info(text)
        if ans is None: # avoid NoneType error
            print("Answer Option Not Extracted")
            ans = next(key for key, value in mapping.items() if value == "ask more questions")
        return rsn, ques, ans

        # self.result_one_round = {
        #     "scene": self.metadata["scene"],
        #     "seq": self.metadata["seq"],
        #     "pair": self.metadata["pair"],
        #     "dof": self.metadata["significance"],
        #     "label text": self.metadata["significance_text"],
        #     "label value": self.metadata["significance_value"],
        #     "round idx": self.round_num,
        #     "pred option": self.ans,
        #     "pred text": self.option_map[self.ans],
        #     "reason": self.rsn,
        #     "question": self.ques,
        # }

        # self.result_dict.append(self.result_one_round)

    def __call__(self, conclusion_from_LLM: str, mapping: dict) -> dict:
        rsn, ques, ans = self._process_conclusion(conclusion_from_LLM, mapping)
        result = {
            "pred option": ans,
            "pred text": mapping[ans],
            "reason": rsn,
            "question": ques,
        }
        return result
    
    def _evaluate(self):
        results = Stat0123Conv(self.result_dict)._calculate_metrics()
        return results


class Stat0123Conv:

    def __init__(self, result_dict):

        self.data = pd.DataFrame(result_dict)

        self.stat = {}

    def _summary_stat(self):

        # find the last round for each pair
        summary = self.data.loc[self.data.groupby(['scene', 'seq', 'pair'])['round idx'].idxmax()]

        total_samples = len(summary)
        count_zero = len(summary[summary["pred text"] == "ask more questions"])
        zero_percentage = count_zero / total_samples * 100

        filtered_data = summary[summary["pred text"] != "ask more questions"]

        y_true = filtered_data["label text"]
        y_pred = filtered_data["pred text"]

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, and F1 Score (for multi-class classification)
        precision = precision_score(y_true, y_pred, pos_label="leftward", average="weighted")
        recall = recall_score(y_true, y_pred, pos_label="leftward", average="weighted")
        f1 = f1_score(y_true, y_pred, pos_label="leftward", average="weighted")

        # Confusion Matrix
        labels = ["leftward", "rightward", "no movement"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        # Return the metrics as a dictionary
        scalar_metrics = {
            "length of dataset": [total_samples],  # Total samples
            "ask more questions percentage": [zero_percentage],  # Make sure you have this value defined
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
        }

        metrics = {
            "scalar metrics": scalar_metrics,
            "confusion matrix": cm_df
        }

        return metrics

    def _calculate_metrics(self):

        metrics = self._summary_stat()
        self.stat[f"summary stat"] = metrics

        max_round = self.data["round idx"].max()

        for round in range(1, max_round+1):

            round_stat = self.data[self.data["round idx"] == round]

            total_samples = len(round_stat)
            count_zero = len(round_stat[round_stat["pred text"] == "ask more questions"])
            zero_percentage = count_zero / total_samples * 100

            filtered_data = round_stat[round_stat["pred text"] != "ask more questions"]

            y_true = filtered_data["label text"]
            y_pred = filtered_data["pred text"]

            # Accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Precision, Recall, and F1 Score (for multi-class classification)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            # Confusion Matrix
            labels = ["leftward", "rightward", "no movement"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            # Return the metrics as a dictionary
            scalar_metrics = {
                "length of dataset": [total_samples],
                "ask more questions percentage": [zero_percentage],
                "accuracy": [accuracy],
                "precision": [precision],
                "recall": [recall],
                "f1_score": [f1],
            }

            metrics = {
                "scalar metrics": scalar_metrics,
                "confusion matrix": cm_df
            }

            self.stat[f"{round} round stat"] = metrics

        return self.stat
