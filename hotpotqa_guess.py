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
from os.path import join

class HotPotQARun:
    def __init__(self):
        openai.api_key = Constants.openai_api_key
        self.gemini_client = genai.Client(api_key=Constants.gemini_api_key)
        self.env = self.get_env()
        self.simulation_observations_dict = {}
        self.current_index = None
        self.base_path = "./trajs"

    def get_env(self):
        env = wikienv.WikiEnv()
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
        print(text)

        if save_log:
            log_path = join(self.base_path, str(self.current_index), "log.txt")
            Utils.append_file(text, log_path)


    def openai_llm(prompt, stop=["\n"]):
        response = openai.Completion.create(
        model=Constants.openai_model_name,
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
        return response["choices"][0]["text"]

    def gemini_llm(self, prompt, stop=["\n"]):
        config = types.GenerateContentConfig(stop_sequences=stop)
        response = self.gemini_client.models.generate_content(model=Constants.gemini_model_name, contents=prompt, config=config)
        return str(response.text)

    def step(self, env, action, simulate=False):
        if simulate:
            obs, r, done, info = env.step(action, step_type="simulate")
            return env.sim_obs, r, done, info
        attempts = 0
        while attempts < 10:
            try:
                return env.step(action)
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

    def generate_thought_actions(self, i, running_prompt, n_calls_badcalls):
        n_calls_badcalls[0]+=1
        thought_action = self.gemini_llm(running_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        thought, action = self.separate_thought_and_action(i, thought_action)
        if action is None:
            self.log("error", 'ohh...', thought)
            n_calls_badcalls[0]+=1
            n_calls_badcalls[1]+=1
            temp_action = self.gemini_llm(running_prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            action = self.extract_action(temp_action)
            if action is None:
                print("Temp Action",temp_action)
                raise ValueError("Action not found in llm output")            
        return thought, action     

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

            thought, action = self.generate_thought_actions(i, running_prompt, n_calls_badcalls) # llm call inside
            if simulate:
                sim_thought, sim_action = self.generate_thought_actions(i, sim_running_prompt, n_calls_badcalls) # llm call inside
                sim_running_prompt = running_prompt
            
            # normal trajectory
            obs, r, done, info = self.step(self.env, self.action_lowercase(action))
            obs = obs.replace('\\n', '')
            self.env.update_traj_dict_records(thought, action, obs, False)
            next_step_string = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            running_prompt += next_step_string
            if to_print:
                self.log("NORMAL TRAJECTORY:\n", next_step_string.strip())

            if simulate:
                sim_obs, sim_r, sim_done, sim_info = self.step(self.env, self.action_lowercase(action), simulate=True)
                sim_obs = sim_obs.replace('\\n', '')
                self.env.update_traj_dict_records(sim_thought, sim_action, sim_obs, True)
                next_sim_step_string = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {sim_obs}\n"
                if to_print:
                    self.log("SIMULATION TRAJECTORY:\n", f"Thought {i}: {sim_thought}\nAction {i}: {sim_action}\nObservation {i}: {sim_obs}\n")
                sim_running_prompt += next_sim_step_string
            self.log("\n")
            if done:
                break
        if not done:
            obs, r, done, info = self.step(self.env, "finish[]")
        if to_print:
            self.log("info", info, '\n')
        info.update({'n_calls': n_calls_badcalls[0], 'n_badcalls': n_calls_badcalls[1], 'traj': running_prompt})
        return r, info        

    def run(self, webthink_simulate=False, skip_done=False):
        idxs = list(range(Constants.num))
        random.Random(Constants.random_seed).shuffle(idxs)
        prompt_path = join(Constants.prompts_folder, Constants.prompt_file)
        prompt_dict = Utils.read_json(prompt_path)
        webthink_examples = prompt_dict['webthink_simple6']
        webthink_prompt = Utils.join_prompt(Constants.react_instruction, webthink_examples, Constants.prompt_instruction)

        rewards = []
        infos = []
        old_time = time.time()
        for i in idxs[:Constants.n_samples_to_run]:
            self.current_index = i
            current_dir_path = join(self.base_path, str(self.current_index))
            log_path = join(current_dir_path, "log.txt")
            if skip_done:
                if os.path.exists(log_path):
                    self.log("INFO", f"Skipping index {i} because it is already done", save_log=False)
                    continue
            Utils.delete_file(log_path)
            try:
                r, info = self.webthink(i, prompt=webthink_prompt, to_print=True, n=Constants.n_steps_to_run, simulate=webthink_simulate)
            except ClientError as e:
                self.log("info", "Client Error!! Sleeping for 30 seconds...", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(30)
                continue
            except ServerError as e:
                self.log("info", "Server Error!! Sleeping for 60 seconds...", save_log=False)
                Utils.delete_dir(current_dir_path, nested=True)
                time.sleep(60)
                continue
            rewards.append(info['em'])
            infos.append(info)
            self.log("info", "Sum of rewards:", sum(rewards),"\nNumber of steps:", len(rewards), "\nAverage reward:", sum(rewards) / len(rewards), "\nAverage time", (time.time() - old_time) / len(rewards))
            self.log('-------------------------------------------------------')

            save_dir = join(self.base_path, str(i))

            normal_observations_dict = self.env.normal_trajectory_dict
            sim_observations_dict = self.env.sim_trajectory_dict
            Utils.save_json(normal_observations_dict, join(save_dir, "normalobs.json"))
            Utils.save_json(sim_observations_dict, join(save_dir, "simobs.json"))

            metric_dict = Metrics.get_action_specific_metrics(normal_observations_dict, sim_observations_dict, sparse=False)
            self.log("INFO", f"Index {self.current_index} actions metrics: \n", json.dumps(metric_dict, indent=4), save_log=False)
            Utils.save_json(metric_dict, join(save_dir, "metrics.json"))
        

if __name__=="__main__":
    hotpotqa_wiki_runner = HotPotQARun()
    hotpotqa_wiki_runner.run(webthink_simulate=True, skip_done=True)

    avg_actions_metric, n_samples = Metrics.get_avg_metric("./trajs")
    print("AVERAGE METRIC:", avg_actions_metric, f"for {n_samples} observations")