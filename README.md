# BDD-Bench evaluation

This repository evaluates coding agents on published BDD-Bench instances.
Each instance is a digest-pinned Docker image plus the patches and metadata needed to replay its test environment.

## Get the source code

Download and unpack the anonymous repository snapshot, then enter it:

```bash
mkdir bdd-bench-evaluation
curl -L -o bdd-bench-evaluation.zip \
  https://anonymous.4open.science/api/repo/BDD-Bench-3B58/zip
unzip -q bdd-bench-evaluation.zip -d bdd-bench-evaluation
cd bdd-bench-evaluation
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker, with permission to run containers
- Credentials for the model provider you choose (not needed for `dummy` or `golden`)

Install the evaluator:

```bash
uv sync
cp .env.example .env  # optional: add credentials for your chosen provider
```

The supplied Docker images are public. No GitHub or Docker registry credentials are needed.

## Unpack evaluation data

The source snapshot includes the versioned data archive. Verify it before unpacking:

```bash
sha256sum -c evaluation-data-v1.tar.gz.sha256
tar -xzf evaluation-data-v1.tar.gz
```

The archive creates:

```text
output_dataset/
├── dataset_instances.json
└── evaluation_artifacts/
```

The manifest pins every Docker image by digest. Keep the archive and its artifacts together.

## Verify the evaluator

Start with one published instance. `dummy` submits an empty patch and checks the normal
evaluation path. `golden` submits the reference code patch and should resolve the instance.

```bash
INSTANCE_ID=jrnl-org-jrnl-chain-1-stage-1-initial-99c19b2a64d4-final-cd865e048ee6

PYTHONPATH=src uv run python -m bdd_bench.evaluation.harness \
  --agent dummy --instance-id "$INSTANCE_ID" --generate-and-evaluate

PYTHONPATH=src uv run python -m bdd_bench.evaluation.harness \
  --agent golden --instance-id "$INSTANCE_ID" --generate-and-evaluate
```

Run the same configuration against a model by supplying its normal provider credentials and
model name:

```bash
PYTHONPATH=src uv run python -m bdd_bench.evaluation.harness \
  --model openai/gpt-5-mini \
  --instance-id "$INSTANCE_ID" \
  --generate-and-evaluate
```

Remove `--instance-id` to run the full published benchmark. Model runs are replications of
the protocol, not exact-score checks: hosted models and agent sampling are not deterministic.
Every run writes its resolved settings to `output_evaluation/<run-id>/config.json`.

## Configuration

Copy `.env.example` to `.env` only when environment variables are more convenient than CLI
flags. The template contains evaluation-provider settings only. For a `mini-swe` model, use
the credential variable required by its provider (for example `OPENAI_API_KEY`) and select a
model with `--model` or `BDD_BENCH_MINI_SWE_MODEL`.

The evaluator creates isolated task containers: they have no network, host bind mounts, Linux
capabilities, or privilege escalation. Model credentials stay in the host controller.
