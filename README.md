# OpenJev

**Fast, calibrated, typed decisions from an open model.** OpenJev is an open-source
"System One" decision server: send it a state and a set of typed questions (yes/no,
choice, score) and get back probabilities and a confidence for every answer, in tens of
milliseconds. There is no text generation and no parsing, so the answers cannot be
hallucinated off-schema.

It speaks the same wire API as TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
so their SDKs work against it unchanged. It runs on
[DiffusionGemma 26B-A4B](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)
(Apache-2.0) through vLLM.

> **Hosted for free on [Codiv](https://codiv.ai)**, an inference platform for open System One
> models: sign up and get 100M input tokens, no card required. `https://api.codiv.ai/v1/systemone`

OpenJev is an independent project. It is not affiliated with or endorsed by TypeSafe AI.

## Try it

```bash
pip install typesafe-sdk
export TYPESAFE_BASE_URL=https://api.codiv.ai   # or http://127.0.0.1:8080 for your own server
export TYPESAFE_API_KEY=sk-codiv-...
```

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
r = client.system_one(
    "Everything is down and we have a demo with our biggest client at noon.",
    {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "team":   {"type": "choice", "instructions": "Which team should handle it?",
                   "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
        "tone":   {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]},
    },
)
r.nouls["urgent"].noul        # 1.00
r.choices["team"].choice      # "outage", confidence 1.00
r.scores["tone"].score        # 2.00 (expected level, 0-indexed)
```

Or with curl:

```bash
curl https://api.codiv.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openjev-latest", "state": "I was charged twice this month.",
       "questions": {"is_billing": {"type": "noul", "instructions": "Is this a billing issue?"}}}'
```

## API

| | |
|---|---|
| `POST /v1/systemone` | `{state, model, questions}` → `{model, answers, usage}` |
| `GET /v1/models` | `openjev-0.1` and its alias `openjev-latest`. `jev-latest` and `jev-preview` are also accepted, so TypeSafe SDK defaults work. |

Question types:

- **`noul`** (yes/no): takes optional `criteria: {true, false}` and returns `{noul: P(yes)}`.
- **`choice`**: takes `criteria: {name: description}` and returns `{choice, probabilities, confidence}`.
- **`score`**: takes `criteria: [level0, level1, …]` (2–10 levels) and returns `{score: Σ i·pᵢ, legend, probabilities, confidence}`.

`confidence` is `1 − H(p)/ln K`. It is 1 when the model is certain and 0 when the distribution is uniform.
`usage.input_tokens` counts prompt tokens; output tokens are always 0. Errors follow the
same shapes as Jev: FastAPI `422` validation lists, `{"detail": {"error_type", "message"}}` for
auth errors (`401`/`403`), `429`, and `529` when overloaded.

Known differences from Jev:
- A choice can have at most 128 options (Jev allows 255).
- English only.
- Many questions are answered in chunks of about 12 per read. They are still answered in parallel.

## How it works

DiffusionGemma is a discrete diffusion model: it denoises a whole canvas of tokens per
forward pass instead of generating left to right. OpenJev writes the answer template
onto the canvas, for example:

```
q1: ▒
q2: ▒
q3: ▒
```

Only the label slots are left as noise. One read-only denoise step then gives a full
probability distribution over each question's labels, and those distributions are the
answers. If any slot is uncertain (entropy > 0.1), OpenJev re-reads with fresh noise up to
four times and averages the results. Question ids never reach the model.

The vLLM side of this is [vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250),
which adds seeded canvases, read-only steps and step caps for DiffusionGemma. `openjev/engine.py` is
adapted from that PR's `structured_server.py` example, with async I/O, bounded concurrency and
backpressure added.

## Run your own

You need an NVIDIA GPU with at least 24 GB of memory for the NVFP4 checkpoint (tested on an RTX PRO 6000 Blackwell, sm_120).

Prebuilt images are on Docker Hub, so there is nothing to compile:

| Image | Contents |
|---|---|
| [`razorback16/openjev`](https://hub.docker.com/r/razorback16/openjev) | The Jev-compatible API server (small) |
| [`razorback16/openjev-vllm`](https://hub.docker.com/r/razorback16/openjev-vllm) | vLLM with PR #57250 at `d2c2b54`, CUDA 13 (large) |

```bash
git clone https://github.com/razorback16/openjev && cd openjev
docker compose up -d          # pulls both images; OpenJev on 127.0.0.1:8080
curl localhost:8080/v1/models
```

The model weights (about 18 GB) download on first start into `~/.cache/huggingface`.
Use `docker compose build` to build the images yourself instead.

Settings are read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_UPSTREAM` | `http://127.0.0.1:8000` | vLLM server URL |
| `OPENJEV_CANVAS` | `64` | canvas length; must match vLLM's `--diffusion-config` |
| `OPENJEV_MAX_INFLIGHT` | `64` | reads in flight to vLLM |
| `OPENJEV_MAX_QUEUE` | `512` | waiting decisions before the server returns 529 |
| `OPENJEV_API_KEY` | unset | require `Authorization: Bearer <key>` |
| `OPENJEV_ORIGIN_SECRET` | unset | require an `X-Origin-Secret` header (for use behind a proxy) |

Measured on an RTX PRO 6000 using 38% of the GPU, with 3 questions per request and cache-busted states:

| Concurrency | req/s | p50 | p95 |
|---:|---:|---:|---:|
| 1 | 10.7 | 94 ms | 94 ms |
| 16 | 43.3 | 367 ms | 369 ms |
| 32 | 51.7 | 545 ms | 618 ms |
| 64 | 57.4 | 760 ms | 1109 ms |

Without Docker:

```bash
git clone https://github.com/mmastrac/vllm -b structured-reads-main && cd vllm
git checkout d2c2b5422d && VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=2c88fb131c7ae0be01907cd8c276911db5e7aad4 pip install -e .
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 --served-model-name dgemma \
  --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching \
  --async-scheduling --attention-backend TRITON_ATTN
pip install -e path/to/openjev && python -m openjev
```

## Caveats

- vllm-project/vllm#57250 has not been merged. The request fields it uses (`vllm_xargs`) are
  provisional, so this project pins the fork at commit `d2c2b54`.
- Answer quality is the quality of DiffusionGemma 26B-A4B used in this mode. Evaluate it on your
  own tasks before relying on it.

## Development

```bash
pip install -e '.[test]' && pytest
```

## License

Apache-2.0. The DiffusionGemma weights are Apache-2.0 (NVIDIA / Google).
