import os
import json
from utils import Utils

class Metrics:

    @staticmethod
    def get_actions_metric(dict1, dict2, sparse=False):
        actions1 = dict1["actions"]
        actions2 = dict2["actions"]
        assert len(actions1) == len(actions2), "Different number of actions taken in wiki and guess"
        n = len(actions1)
        score = 0
        for a1, a2 in zip(actions1, actions2):
            score+=Metrics.compare_action(a1, a2, sparse)
        return score/n

    @staticmethod
    def get_action_specific_metrics(normal_dict, sim_dict, sparse=False):
        normal_actions = normal_dict["actions"]
        sim_actions = sim_dict["actions"]
        assert len(normal_actions) == len(sim_actions), "Different number of actions taken in wiki and guess"
        metric_dict = {"general": 0}
        count_dict = {"general": 0}
        for na, sa in zip(normal_actions, sim_actions):

            metric_dict["general"] += Metrics.compare_action(na, sa, sparse)
            count_dict["general"] += 1

            na_name = Metrics.get_action_name(na)
            sa_name = Metrics.get_action_name(sa)
            if na_name in metric_dict.keys():
                metric_dict[na_name] += Metrics.compare_action(na, sa, sparse)
                count_dict[na_name] += 1
            else:
                metric_dict[na_name] = Metrics.compare_action(na, sa, sparse)
                count_dict[na_name] = 1
        
        for key in metric_dict.keys():
            metric_dict[key]/= count_dict[key]
        return metric_dict
        
        
    
    @staticmethod
    def compare_action(action1, action2, sparse=False):
        action1 = action1.lower()
        action2 = action2.lower()
        if sparse:
            if Metrics.get_action_name(action1) == Metrics.get_action_name(action2):
                return 1
            else:
                return 0
        else:            
            if action1 == action2:
                return 1
            else:
                return 0
    
    @staticmethod
    def get_action_name(full_action):
        full_action = full_action.strip()
        ind = full_action.find('[')
        if ind < 0:
            return None
        else:
            return full_action[:ind]
    
    @staticmethod
    def get_action_specific_avg_metric(dir_path):
        avg_metric = 0
        n_metrics = 0
        for direc in os.listdir(dir_path):
            metrics_dict = Utils.read_json(os.path.join(dir_path, direc, "metrics.json"))
            metric = metrics_dict["general"]
            avg_metric += metric
            n_metrics+=1
        avg_metric/=n_metrics
        return avg_metric, n_metrics
        