# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict
from copy import deepcopy
import torch
from recbole.utils import dict2str, set_color
from logging import getLogger


class Evaluator(object):
    """
    Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    It now supports evaluating popular and unpopular user groups separately.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}
        self.HT_ratio = config['HT_ratio']
        self.divide = config['divide']
        self.logger = getLogger()

        # 初始化每個指標對應的計算類
        for metric in self.metrics:
            if metric not in metrics_dict:
                raise ValueError(f"Metric '{metric}' is not supported.")
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def _split_data_by_user_popularity(self, dataobject: DataStruct):
        if "data.label" in dataobject._data_dict:
            label_data = dataobject.get("data.label")
            user_interaction_counts = label_data.sum(dim=1)  # 每個用戶的交互數
        elif "data.count_users" in dataobject._data_dict:
            user_interaction_counts = dataobject.get("data.count_users")
        elif "rec.topk" in dataobject._data_dict:
            # 從 rec.topk 中推導交互次數
            rec_topk = dataobject.get("rec.topk")
            user_interaction_counts = rec_topk[:, -1]  # 最後一列是正確項目數
        else:
            print("Warning: Could not find `data.label`, `data.count_users`, or derive it from `rec.topk`. Returning original data.")
            return deepcopy(dataobject), deepcopy(dataobject)

        n_users = user_interaction_counts.size(0)
        top_10_percent = int(n_users * self.HT_ratio)
        _, sorted_indices = torch.sort(user_interaction_counts, descending=True)
        popular_user_indices = sorted_indices[:top_10_percent]
        unpopular_user_indices = sorted_indices[top_10_percent:]

        popular_dataobject = deepcopy(dataobject)
        unpopular_dataobject = deepcopy(dataobject)

        for key, value in dataobject._data_dict.items():
            if isinstance(value, torch.Tensor) and value.size(0) == n_users:
                popular_dataobject._data_dict[key] = value[popular_user_indices]
                unpopular_dataobject._data_dict[key] = value[unpopular_user_indices]

        return popular_dataobject, unpopular_dataobject

    def evaluate(self, dataobject: DataStruct):
        """
        Calculate all metrics, optionally split by user popularity.

        Args:
            dataobject (DataStruct): Contains all the information needed for metrics.
            split_by_popularity (bool): Whether to split data into popular and unpopular users.

        Returns:
            collections.OrderedDict: Overall metric results or split results if split_by_popularity is True.
        """
        if self.divide:
            popular_data, unpopular_data = self._split_data_by_user_popularity(dataobject)

            popular_results = self._calculate_metrics(popular_data)

            unpopular_results = self._calculate_metrics(unpopular_data)

            result_dict = OrderedDict()
            result_dict["head"] = popular_results
            result_dict["tail"] = unpopular_results

            return result_dict
        else:
            result_dict = OrderedDict()
            for metric in self.metrics:
                metric_val = self.metric_class[metric].calculate_metric(dataobject)
                result_dict.update(metric_val)
            valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(result_dict)
                )
        return result_dict

    def _calculate_metrics(self, dataobject: DataStruct):
        """
        Calculate metrics for a single DataStruct.

        Args:
            dataobject (DataStruct): DataStruct containing evaluation data.

        Returns:
            collections.OrderedDict: Metric results for the provided DataStruct.
        """
        group_result = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            group_result.update(metric_val)
        return group_result
