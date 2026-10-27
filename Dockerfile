# The host driver is 570 (CUDA 12.8), so it runs via CUDA minor-version compatibility:
#  - NVIDIA_DISABLE_REQUIRE skips nvidia-container-cli's "cuda>=12.9" check;
#  - the cuda-compat (forward compatibility) libs are removed because they only
#    work on datacenter GPUs (Error 804 on RTX 3090), so the host libcuda is used.
FROM vllm/vllm-openai:v0.30.0-cu129
ENV NVIDIA_DISABLE_REQUIRE=1
RUN rm -rf /usr/local/cuda-*/compat && ldconfig
# This image only ships python3; the run scripts call `python`.
RUN command -v python >/dev/null || ln -s "$(command -v python3)" /usr/local/bin/python

# Install project dependencies
# Pin the packages vLLM ships with, so project deps can't upgrade them
# (e.g. a newer transformers can break vLLM at import time).
COPY requirements.txt /tmp/requirements.txt
RUN pip freeze | grep -iE '^(transformers|tokenizers|huggingface-hub|torch|torchvision|torchaudio|numpy|safetensors|xformers|triton|pydantic|fastapi|openai)==' \
      > /tmp/vllm-constraints.txt \
 && pip install --no-cache-dir -c /tmp/vllm-constraints.txt -r /tmp/requirements.txt

# Keep the executable benchmark and llmwiki2's benchmark prompts in the image.
WORKDIR /workdir
COPY benchmark_llm_wiki.py /workdir/benchmark_llm_wiki.py
COPY prompts_benchmark /workdir/prompts_benchmark
