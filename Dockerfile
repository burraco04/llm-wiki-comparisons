FROM vllm/vllm-openai:latest

# Install project dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Keep the executable benchmark and llmwiki2's benchmark prompts in the image.
WORKDIR /workdir
COPY benchmark_llm_wiki.py /workdir/benchmark_llm_wiki.py
COPY prompts_benchmark /workdir/prompts_benchmark
