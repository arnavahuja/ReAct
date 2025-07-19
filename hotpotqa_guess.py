import os
import openai
from google import genai
from google.genai import types
import wikienv, wrappers
import requests
import json
import sys
import random
import time
from constants import *

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

 
openai.api_key = os.environ.get("OPENAI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)

def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]

def gemini_llm(prompt, stop=["\n"]):
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0),stop_sequences=stop)
    response = gemini_client.models.generate_content(model=gemini_model_name, contents=prompt, config=config)
    return response.text


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1




folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True, n=8):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, n):
        n_calls += 1
        thought_action = gemini_llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except Exception as e:
            print(e)
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = gemini_llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info        



idxs = list(range(7405))
random.Random(238).shuffle(idxs)

rs = []
infos = []
old_time = time.time()
for i in idxs[:5]:
    r, info = webthink(i, to_print=True)
    rs.append(info['em'])
    infos.append(info)
    print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    print('-----------')
    print()