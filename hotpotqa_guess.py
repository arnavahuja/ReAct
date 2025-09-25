import os
import openai
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
import wikienv, wrappers
import requests
import json
import sys
import random
import time
import constants as Constants
from utils import Utils
import logging
import re
from metrics import Metrics
from grapher import Grapher
from os.path import join
import argparse
from openai import OpenAI
from openai._exceptions import OpenAIError
from prompts import PromptTemplates

class HotPotQARun:
    def __init__(self, model_name="gemini", guess_model_name="gemini", to_print_output=True):
        openai.api_key = Constants.openai_api_key
        self.gemini_client = genai.Client(api_key=Constants.gemini_api_key)
        self.openai_client = openai.OpenAI(api_key=Constants.openai_api_key)
        self.openrouter_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=Constants.openrouter_api_key)
        self.model_name = model_name
        self.guess_model_name = guess_model_name
        self.env = self.get_env()
        self.simulation_observations_dict = {}
        self.current_index = None
        self.base_traj_path = self.recalc_base_traj_path()
        self.to_print_output = to_print_output

    def recalc_base_traj_path(self):
        btp =  "./run_metrics/agent_"+ self.model_name.split('/')[-1] + "_top" + str(Constants.guess_num_actions) + "/trajs_" + self.guess_model_name.split("/")[-1] 
        self.base_traj_path = btp
        return btp

    def get_env(self):
        env = wikienv.WikiEnv(guess_model_name=self.guess_model_name)
        env = wrappers.HotPotQAWrapper(env, split="dev")
        env = wrappers.LoggingWrapper(env)
        env = wrappers.HistoryWrapper(env, obs_format="history")        
        return env
    
    def log(self, level, *args, save_log=True):
        text = level.upper() + " "
        for arg in args:
            text += str(arg)
            text += " "
        text = text.strip()
        text += "\n"
        if self.to_print_output:
            print(text)

        if save_log:
            log_path = join(self.base_traj_path, str(self.current_index), "log.txt")
            Utils.append_file(text, log_path)


    def openai_llm(self, prompt, stop=["\n"]):
        response = self.openai_client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=Constants.temperature,
        max_tokens=Constants.max_output_tokens,
        top_p=Constants.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def gemini_llm(self, prompt, stop=["\n"]):
        config = types.GenerateContentConfig(stop_sequences=stop)
        response = self.gemini_client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        return str(response.text)
    
    def openrouter_llm(self, prompt, stop=["\n"]):
        response = self.openrouter_client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=Constants.temperature,
        max_tokens=Constants.max_output_tokens,
        top_p=Constants.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        )
        return response.choices[0].message.content
    
    def call_llm(self, prompt, stop=["\n"]):
        if self.model_name.startswith("gemini"):
            return self.gemini_llm(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self.openai_llm(prompt, stop)
        else:
            return self.openrouter_llm(prompt, stop)

    def step(self, env, action, simulate=False):

        if simulate:
            start = time.perf_counter()
            obs, r, done, info = env.step(action, step_type="simulate")
            end = time.perf_counter()
            return env.sim_obs, r, done, info, end-start
        
        attempts = 0
        while attempts < 10:
            try:
                start = time.perf_counter()
                obs, r, done, info = env.step(action)
                end = time.perf_counter()
                return obs, r, done, info, end-start
            except requests.exceptions.Timeout:
                attempts += 1

    def extract_action(self, action_string):
        search_pattern = r'[Ss]earch\[[^\]]+\]'
        lookup_pattern = r'[Ll]ookup\[[^\]]+\]'
        finish_pattern = r'[Ff]inish\[[^\]]+\]'
        arr = [search_pattern, lookup_pattern, finish_pattern]
        output = None
        for pattern in arr:
            output = re.search(pattern, action_string)
            if output:
                break
        if output:
            return output.group()
        else:
            return None
        
    def extract_actions(self, action_string):
        search_pattern = r'[Ss]earch\[[^\]]+\]'
        lookup_pattern = r'[Ll]ookup\[[^\]]+\]'
        finish_pattern = r'[Ff]inish\[[^\]]+\]'
        arr = [search_pattern, lookup_pattern, finish_pattern]
        outputs = []

        for pattern in arr:
            current_pattern_output = re.findall(pattern, action_string)
            outputs += current_pattern_output

        return outputs
    
    def action_lowercase(self, action):
        index = action.find("[")
        if index == -1:
            return action[0].lower() + action[1:]
        else:
            return action[:index].lower() + action[index:]
    
    def separate_thought_and_action(self, i, thought_action):
        action_condition = f"Action {i}: " in thought_action
        thought_condition = f"Thought {i}: " in thought_action
        thought, action = None, None
        if thought_condition and action_condition:
            thought, action = thought_action.strip().split(f"Action {i}: ")
            thought = thought.strip().split(f"Thought {i}: ")[1]
        elif thought_condition:
            thought = thought_action.strip().split(f"Thought {i}: ")[1]
            action = None
        elif action_condition:
            action = thought_action.strip().split(f"Action {i}: ")[1]
            thought = "Let me do the action " + action
        return thought, action

    def separate_thought_and_actions(self, i, thought_action):
        try:
            split = thought_action.split(f"Thought {i}: ")
            thought = split[0].strip() if len(split) == 1 else split[1].strip()
        except IndexError:
            thought = thought_action.strip()
        actions = self.extract_actions(thought_action)
        return thought, actions
    

    def generate_thought_actions(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        n_calls_badcalls[0]+=1
        
        thought_action = self.call_llm(running_prompt + PromptTemplates.ACTION_GUESS_PROMPT.format(i=i, num_guesses=num_actions), stop=None)
        self.log("LLM RESPONSE", thought_action.strip(), save_log=False)
        thought, actions = self.separate_thought_and_actions(i, thought_action)

        retry_attempt = 1
        while not actions and retry_attempt <= max_retries:
            self.log("error", 'Actions not found in', thought_action, "\nRetrying to get actions...")
            n_calls_badcalls[0]+=1
            n_calls_badcalls[1]+=1
            temp_actions = self.call_llm(running_prompt + PromptTemplates.RETRY_PROMPT.format(attempt=retry_attempt, role=Constants.agent_role, num_guesses=num_actions))
            actions = self.extract_actions(temp_actions) 
            
            retry_attempt += 1

        if not actions:
            self.log("ERROR", "No action found", "Action String", actions, save_log=False)
            raise ValueError("Action not found in llm output")  
                     
        return thought, actions     

    def webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        done = False
        running_prompt = prompt
        question = self.env.reset(idx=idx)
        running_prompt += question + "\n"
        self.env.normal_trajectory_dict["prompt"] = running_prompt

        if to_print:
            self.log("info", idx, question)

        if simulate:
            sim_running_prompt = running_prompt
            self.env.sim_trajectory_dict["prompt"] = sim_running_prompt

        n_calls_badcalls = [0, 0]

        for i in range(1, n):
            if to_print:
                self.log(f"STEP: {i}")
            try:
                thought, actions = self.generate_thought_actions(i, running_prompt, n_calls_badcalls, max_retries=Constants.max_agent_retries) # llm call inside
                action = actions[0]
                if simulate:
                    sim_thought, sim_actions = self.generate_thought_actions(i, sim_running_prompt, n_calls_badcalls, num_actions=Constants.guess_num_actions, max_retries=Constants.max_guess_retries) # llm call inside
                    sim_running_prompt = running_prompt
            except ValueError as e:
                self.log("error", e, save_log=False)
                continue
            
            # normal trajectory
            obs, r, done, info, normal_traj_time_taken = self.step(self.env, self.action_lowercase(action))
            obs = obs.replace('\\n', '')
            self.env.update_traj_dict_records(thought, action, obs, normal_traj_time_taken, False)
            next_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(i=i, thought=thought, action=action, obs=obs)
            running_prompt += next_step_string
            if to_print:
                self.log("NORMAL TRAJECTORY:\n", next_step_string.strip())

            if simulate:
                sim_obs, sim_r, sim_done, sim_info, sim_traj_time_taken = self.step(self.env, self.action_lowercase(action), simulate=True)
                sim_obs = sim_obs.replace('\n', '')
                self.env.update_traj_dict_records(sim_thought, sim_actions, sim_obs, sim_traj_time_taken, True)
                next_sim_step_string = PromptTemplates.NEXT_STEP_PROMPT.format(i=i, thought=thought, action=action, obs=sim_obs)
                if to_print:
                    self.log("SIMULATION TRAJECTORY:\n", PromptTemplates.NEXT_STEP_PROMPT.format(i=i, thought=sim_thought, action=sim_actions, obs=sim_obs))
                sim_running_prompt += next_sim_step_string
            self.log("\n")
            if done:
                break
        if not done:
            obs, r, done, info, time_taken = self.step(self.env, "finish[]")
        if to_print:
            self.log("info", info, '\n')
        info.update({'n_calls': n_calls_badcalls[0], 'n_badcalls': n_calls_badcalls[1], 'traj': running_prompt})
        return info        

    def run(self, webthink_simulate=False, skip_done=False):

        idxs = list(range(Constants.num))
        random.Random(Constants.random_seed).shuffle(idxs)

        webthink_examples = Utils.read_json(join(Constants.prompts_folder, Constants.prompt_file)).get('webthink_simple6')
        webthink_prompt = Utils.join_prompt(PromptTemplates.REACT_INSTRUCTION, webthink_examples, PromptTemplates.PROMPT_INSTRUCTION)

        rewards = []
        infos = []
        old_time = time.time()

        for i in idxs[:Constants.n_samples_to_run]:
            self.current_index = i
            current_dir_path = join(self.base_traj_path, str(self.current_index))

            log_path = join(current_dir_path, "log.txt")
            if skip_done:
                if os.path.exists(log_path):
                    self.log("INFO", f"Skipping index {i} because it is already done", save_log=False)
                    continue

            Utils.delete_file(log_path)

            try:
                info = self.webthink(i, prompt=webthink_prompt, to_print=True, n=Constants.n_steps_to_run, simulate=webthink_simulate)

            except ClientError as e:
                self.log("info", f"Client Error!! Sleeping for {Constants.client_error_sleep_time} seconds...", save_log=False)
                self.log("error",e,save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(Constants.client_error_sleep_time)
                continue

            except ServerError as e:
                self.log("info", f"Server Error!! Sleeping for {Constants.server_error_sleep_time} seconds...", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(Constants.server_error_sleep_time)
                continue
            
            except ZeroDivisionError as e:
                self.log("info", "ZeroDivisionError!!", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue

            except Exception as e:
                Utils.delete_dir(current_dir_path, nested=True)
                raise e

            rewards.append(info['em'])
            infos.append(info)

            self.log("info", "Sum of rewards:", sum(rewards),"\nNumber of steps:", len(rewards), "\nAverage reward:", sum(rewards) / len(rewards), "\nAverage time", (time.time() - old_time) / len(rewards))
            self.log('-'*50)

            normal_observations_dict = self.env.normal_trajectory_dict
            sim_observations_dict = self.env.sim_trajectory_dict
            Utils.save_json(normal_observations_dict, join(current_dir_path, "normalobs.json"))
            Utils.save_json(sim_observations_dict, join(current_dir_path, "simobs.json"))
            try:
                metric_dict = Metrics.get_action_specific_metrics(normal_observations_dict, sim_observations_dict, sparse=False)
            except ZeroDivisionError as e:
                self.log("info", "ZeroDivisionError in metrics!!", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                continue
            self.log("INFO", f"Index {self.current_index} actions metrics: \n", json.dumps(metric_dict, indent=4), save_log=False)
            Utils.save_json(metric_dict, join(current_dir_path, "metrics.json"))
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Arguments for running the hotpotqa script")
    
    parser.add_argument("--getmetric",action="store_true", help="Flag to get average metrics printed")
    parser.add_argument("--getmetric2",action="store_true", help="Flag to get average metrics printed")
    parser.add_argument("--savemetrics",action="store_true", help="Flag to get average metrics printed")
    parser.add_argument("--graph",action="store_true", help="Flag to plot graphs")
    parser.add_argument("--graph2",action="store_true", help="Flag to plot comparison graphs")
    parser.add_argument("--graph3",action="store_true", help="Flag to plot comparison graphs")
    parser.add_argument("--noprint", action="store_false", help="Print on command line")
    parser.add_argument("--norun", action="store_false", help="Flag to run all samples")
    parser.add_argument("--modelname",default=Constants.openrouter_model_name, help="Model which the agent will use")
    parser.add_argument("--guessmodelname", default=Constants.openrouter_guess_model_name, help="Model which will be used to guess the wikipedia api call")
    parser.add_argument("--cleanuptrajs", action="store_true", help="Flag to cleanup all trajectories")
    args = parser.parse_args()

    hotpotqa_wiki_runner = HotPotQARun(model_name=args.modelname, guess_model_name=args.guessmodelname, to_print_output=args.noprint)


    if args.cleanuptrajs:
        Utils.cleanup_trajs(hotpotqa_wiki_runner.base_traj_path)
    
    if args.norun:
        hotpotqa_wiki_runner.run(webthink_simulate=True, skip_done=True)
        Utils.cleanup_trajs(hotpotqa_wiki_runner.base_traj_path)
    
    agent_model_names = ["google/gemini-2.5-flash", "openai/gpt-4", "openai/gpt-5"]
    openrouter_guess_model_names = ["openai/gpt-5-nano", "openai/gpt-4.1-nano", "openai/gpt-3.5-turbo", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]

    if args.getmetric:
        all_avg_metrics = {}
        for agent in agent_model_names:
            all_avg_metrics[agent] = {}
            for guess_model in openrouter_guess_model_names:
                hotpotqa_wiki_runner.model_name = agent
                hotpotqa_wiki_runner.guess_model_name = guess_model
                hotpotqa_wiki_runner.recalc_base_traj_path()
                if not os.path.exists(hotpotqa_wiki_runner.base_traj_path):
                    print(f"Path {hotpotqa_wiki_runner.base_traj_path} does not exist. Skipping...")
                    continue
                avg_metrics_dict, n_samples = Metrics.get_action_specific_avg_metric(hotpotqa_wiki_runner.base_traj_path, get_time=True)
                print(f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n", json.dumps(avg_metrics_dict,indent=4), f"\nfor {n_samples} observations")
                all_avg_metrics[agent][guess_model] = avg_metrics_dict
        metrics_df = Utils.convert_json_to_csv(all_avg_metrics)
        if args.savemetrics:
            Utils.save_json(all_avg_metrics, join("./run_metrics", "all_avg_metrics_top{}.json".format(Constants.guess_num_actions)))
            Utils.save_file(metrics_df.to_csv(index=False), join("./run_metrics", "all_avg_metrics_top{}.csv".format(Constants.guess_num_actions)))
    
    if args.getmetric2:
        all_avg_metrics = {}
        for agent in agent_model_names:
            all_avg_metrics[agent] = {}
            for guess_model in openrouter_guess_model_names:
                hotpotqa_wiki_runner.model_name = agent
                hotpotqa_wiki_runner.guess_model_name = guess_model
                hotpotqa_wiki_runner.recalc_base_traj_path()
                if not os.path.exists(hotpotqa_wiki_runner.base_traj_path):
                    print(f"Path {hotpotqa_wiki_runner.base_traj_path} does not exist. Skipping...")
                    continue
                avg_metrics_dict, n_samples = Metrics.cum_metrics(hotpotqa_wiki_runner.base_traj_path)
                print(f"AVERAGE METRICS for agent {agent} using guess model {guess_model}:\n", json.dumps(avg_metrics_dict,indent=4), f"\nfor {n_samples} observations")
                all_avg_metrics[agent][guess_model] = avg_metrics_dict
        metrics_df = Utils.convert_json_to_csv(all_avg_metrics)
        if args.savemetrics:
            Utils.save_json(all_avg_metrics, join("./run_metrics", "list_metrics_top1_3.json"))

    if args.graph:
        data = Utils.read_json(join("./run_metrics", "all_avg_metrics_top{}.json".format(Constants.guess_num_actions)))
        for agent in data.keys():
            Grapher.graph_agent_times(data, agent=agent)
        
    if args.graph2:
        data = Utils.read_json(join("./run_metrics", "comparision_top1_top3.json"))
        Grapher.graph_metric_comparison(data)
    
    if args.graph3:
        data = Utils.read_json(join("./run_metrics", "list_metrics_top1_3.json"))
        Grapher.graph_metric3(data)
        
            