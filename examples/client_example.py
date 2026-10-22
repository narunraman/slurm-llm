"""Start a model on a cluster from Python and query it (no proxy needed).

Run after `sllm init`, `connect`, `setup` and `download qwen-3-8b` (see the README).
"""

from slurm_llm import SlurmLLM

with SlurmLLM("qwen-3-8b", walltime="1:00:00") as llm:
    llm.ensure_ready()  # reuses a running job, or submits one and waits until vLLM is loaded

    reply = llm.generate(messages=[{"role": "user", "content": "What is the capital of France?"}], max_tokens=50)
    print(reply.choices[0].message.content)

    # many requests in parallel; failures come back as exceptions in the list
    questions = [f"What is {n} squared? Answer with the number only." for n in range(5)]
    replies = llm.generate_batch([{"messages": [{"role": "user", "content": q}]} for q in questions], max_tokens=10)
    for question, reply in zip(questions, replies):
        print(question, "->", reply if isinstance(reply, Exception) else reply.choices[0].message.content)

    # llm.cancel_job() would stop the server now; otherwise it runs until its time limit
