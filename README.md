# LLM-Based Guessing to Mitigate API Latency

This project extends the ReAct prompting framework by introducing a novel strategy: **speculative action guessing** using LLMs to reduce end-to-end latency in tool-augmented reasoning.

Instead of blocking computation while waiting for slow API responses (e.g., Wikipedia queries), we *guess* the likely result using an LLM and continue forward execution speculatively. If the guess proves correct, we gain speedup; if not, we can roll back and recompute based on the real API output.

## Motivation

When API calls are the primary bottleneck, compute resources can be leveraged to *trade latency for inference*. This design enables:

- Reduced waiting time in pipelines reliant on external APIs
- Smarter anticipation of next actions when results are weakly coupled to previous actions
- Execution of multiple guesses in parallel to increase robustness

## Potential Applications

- General tool-using agents (e.g., ReAct agents)
- Dependency installation optimization (e.g., pre-fetching Python packages)
- Chatbots (speculatively prepare summaries or responses)
- Search assistants (guess likely queries before user finishes input)
- Simulation platforms (predict human/agent behavior using LLMs)
- Persona simulation (human is treated as a delayed API)
- Predicting API responses (e.g., using tools like [veris.ai](https://veris.ai))

## Features

- LLM-powered speculative reasoning
- Parallel guess execution
- Optional LLM self-assessment on guess quality
- Historical learning of guess reliability for cost-speed tradeoff
- Seamless integration into ReAct-style agent workflows

## Setup

1. Set your OpenAI API key and Gemini API key in the constants.py file

2. Install dependencies:
   ```bash
   pip install openai
   ```

## Experiments

Run the file:`hotpotqa_guess.py`
You can alter the parameters of this run in the constants.py file.


These experiments include speculative guessing of intermediate API/tool outputs. Comparisons are made between original ReAct framework with wikipedia call and speculative call.