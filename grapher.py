import matplotlib.pyplot as plt
import os
import constants as Constants
import numpy as np
from utils import Utils

class Grapher:
    def __init__(self):
        pass
    
    @staticmethod
    def graph_agent_times(data, agent, save_graph=True):
        agent_keys = data[agent].keys()
        guess_models = [key.split('/')[-1] for key in agent_keys]
        normal_times = []
        sim_times = []
        for gm in agent_keys:
            normal_times.append(data[agent][gm]["normal_avg_searchtime"])
            sim_times.append(data[agent][gm]["sim_avg_searchtime"])
        
        assert len(normal_times) == len(sim_times) == len(guess_models)
        width = 0.35
        plt.figure(figsize=(10,5))
        plt.bar(guess_models, sim_times, width=width, label='Sim Avg Search Time')  # overlay bars
        plt.bar(guess_models, normal_times, width=width, label='Normal Avg Search Time')
        plt.ylabel('Time (seconds)')
        plt.title(f'Comparison of Normal vs Sim Avg Search Time for Agent: {agent}')
        plt.xticks(rotation=0)
        plt.legend()
        plt.tight_layout()
        if save_graph:
            filename = "./run_metrics/images/AgentTimes_"+agent.split('/')[-1]+"_top{}".format(Constants.guess_num_actions)+".png"
            if os.path.exists(filename):
                os.remove(filename)
            plt.savefig(filename, dpi=300)
        plt.close()

    
    @staticmethod
    def graph_metric_comparison(data, save_graph=True):
        agents = ["google/gemini-2.5-flash", "openai/gpt-4", "openai/gpt-5"]
        good_dict = {}
        width = 0.35
        for agent in agents:
            good_dict[agent] = {}

        for d in data:
            agent = d["agent_model"]
            good_dict[agent][d["guess_model"]] = d
        
        for agent in good_dict.keys():
            agent_name = agent.split('/')[-1]
            guess_models = []
            top1_accuracies = []
            top3_accuracies = []
            for gm in good_dict[agent].keys():
                top1 = good_dict[agent][gm]["general_top1"]
                top3 = good_dict[agent][gm]["general_top3"]
                if top1 is None:
                    top1 = 0
                if top3 is None:
                    top3 = 0
                guess_models.append(gm.split('/')[-1])
                top1_accuracies.append(top1)
                top3_accuracies.append(top3)

            plt.figure(figsize=(10,5))
            # plt.bar(guess_models - width/2, top1_accuracies, width=width, label='Top 1 Accuracies', alpha=0.5)  # overlay bars
            # plt.bar(guess_models + width/2, top3_accuracies, width=width, label='Top 3 Accuracies', alpha=0.5)
            x = np.arange(len(guess_models))
            plt.bar(x - width/2, top3_accuracies, width=width, label='Top 3 Accuracies')
            plt.bar(x + width/2, top1_accuracies, width=width, label='Top 1 Accuracies')
            plt.ylabel('Accuracy')
            plt.title(f'Comparison of Top 1 and 3 Accuracies for Agent: {agent_name}')
            plt.xticks(x, guess_models, rotation=0)
            plt.legend()
            plt.tight_layout()
            if save_graph:
                filename = "./run_metrics/images/AgentComparison_"+agent_name+".png"
                if os.path.exists(filename):
                    os.remove(filename)
                plt.savefig(filename, dpi=300)
            plt.close()
        
    
    @staticmethod
    def graph_metric3(data, save_graph=True):
        agents = ["google/gemini-2.5-flash", "openai/gpt-4", "openai/gpt-5"]
        good_dict = {}
        width = 0.4
        
        for agent in agents:
            agent_name = agent.split('/')[-1]
            top1_accuracies = []
            top3_accuracies = []
            guess_models = []
            for gm in data[agent].keys():
                top1_list = data[agent][gm]["general_top1"]
                top3_list = data[agent][gm]["general_top3"]
                top1_metric = Utils.avg(top1_list)
                print(top1_metric)
                top3_metric = Utils.avg(top3_list)
                print(top3_metric)
                top1_accuracies.append(top1_metric)
                top3_accuracies.append(top3_metric)
                guess_models.append(gm.split('/')[-1])

            # plt.figure(figsize=(8,8))
            # x = np.arange(len(guess_models))
            # plt.bar(x - width/2, top3_accuracies, width=width, label='Top 3 Accuracies', alpha=0.6)
            # plt.bar(x + width/2, top1_accuracies, width=width, label='Top 1 Accuracies', alpha=0.6)
            # plt.ylabel('Accuracy', fontsize=18)
            # # plt.title(f'Comparison of Top 1 and 3 Accuracies for Agent: {agent_name}', fontsize=16)
            # plt.xticks(x, guess_models, rotation=0, fontsize=18)
            # plt.legend(fontsize=14)
            # plt.tight_layout()
            # plt.grid(True, axis="y", linestyle="--", alpha=0.5)
            # if save_graph:
            #     filename = "./run_metrics/images/AgentComparisonProper_"+agent_name+".png"
            #     if os.path.exists(filename):
            #         os.remove(filename)
            #     plt.savefig(filename, dpi=300)
            # plt.close()

            top3_percentages = [v*100 for v in top3_accuracies]
            top1_percentages = [v*100 for v in top1_accuracies]

            fig, ax = plt.subplots(figsize=(8,6))
            ax.margins(y=0.2)   #
            x = np.arange(len(guess_models))
            bars1 = ax.bar(x - width/2, top3_percentages, width=width, label='Top 3 Accuracies', alpha=0.6)
            bars2 = ax.bar(x + width/2, top1_percentages, width=width, label='Top 1 Accuracies', alpha=0.6)

            ax.set_ylabel('Accuracy', fontsize=20)
            # ax.set_title(f'Comparison of Top 1 and 3 Accuracies for Agent: {agent_name}', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(guess_models, rotation=0, fontsize=20)
            ax.legend(fontsize=16)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)

            # ✅ Add values on top of bars
            ax.bar_label(bars1, labels=[f"{v:.2f}%" for v in top3_percentages], padding=3, fontsize=18)
            ax.bar_label(bars2, labels=[f"{v:.2f}%" for v in top1_percentages], padding=3, fontsize=18)

            plt.tight_layout()

            if save_graph:
                filename = "./run_metrics/images/AgentComparisonProper2_"+agent_name+".png"
                if os.path.exists(filename):
                    os.remove(filename)
                plt.savefig(filename, dpi=300)

            plt.close()
