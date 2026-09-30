"""Use models served by `sllm proxy` with the standard OpenAI client.

    sllm prepare qwen-3-8b
    sllm proxy &
    python examples/openai_api.py --model qwen-3-8b
"""

import argparse

from openai import OpenAI

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="qwen-3-8b", help="a model name from `sllm models`")
parser.add_argument("--url", default="http://localhost:8000/v1")
args = parser.parse_args()

client = OpenAI(base_url=args.url, api_key="unused")  # the proxy does not check keys

messages = [{"role": "user", "content": "Explain what a Slurm job array is in two sentences."}]
reply = client.chat.completions.create(model=args.model, messages=messages, max_tokens=200)
print(reply.choices[0].message.content)

# streaming
messages += [{"role": "assistant", "content": reply.choices[0].message.content},
             {"role": "user", "content": "Now give an example command."}]
for chunk in client.chat.completions.create(model=args.model, messages=messages, max_tokens=200, stream=True):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
