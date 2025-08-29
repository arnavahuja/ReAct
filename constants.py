gemini_api_key = ""
openai_api_key = ""
gemini_model_name = "gemini-2.5-flash"
gemini_guess_model_name = "gemini-2.5-flash"
openai_model_name = "gpt-3.5-turbo"
guess_step_prompt = "Give me information about {} in wikipedia style. If wikipedia information is not available still give me the information. Do not give me any error messages. Just give me the information and nothing else"
m_tokens = 20000
prompts_folder = "./prompts/"
prompt_file = 'prompts_naive.json'
react_instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

prompt_instruction = """Now answer the question. Start with the thought always. Do not under any circumstances give me the action first.
Give me the next thought and action pair based on the prompt"""
random_seed = 248
num = 7405
n_steps_to_run = 8
n_samples_to_run = 100
client_error_sleep_time = 60
server_error_sleep_time = 60
trajectory_filenames = ["log.txt", "normalobs.json", "simobs.json", "metrics.json"]