FROM docker.io/python:3.11.6

WORKDIR /workspace

# Use PDM to keep track of exact versions of dependencies
RUN pip install pdm
COPY pyproject.toml README.md pdm.lock ./
# install dependencies first. PDM also creates a /workspace/.venv here.
ENV PATH="/workspace/.venv/bin:$PATH"
RUN pdm install  --no-self
COPY examples ./examples
COPY funsearch ./funsearch

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers
RUN pip install --no-deps . 
RUN llm install llm-ollama
RUN pip install dm_control

# if running the container
# RUN rm -r ./funsearch ./build
RUN rm -r ./build
CMD /bin/bash

# if debugging
# RUN pip install debugpy
# CMD ["python", "-Xfrozen_modules=off", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "funsearch", "run", "examples/dm_control_ballcup_spec.py", "1", "--sandbox_type", "ExternalProcessSandbox"]
