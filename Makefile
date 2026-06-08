.PHONY: install up down fmt lint test doctor train eval sample demo gpu repro readme summary clean

install:  ## uv sync with dev extras
	uv sync --extra dev

up:  ## bring up MLflow on :5050
	docker compose up -d

down:  ## stop services (volumes preserved)
	docker compose down

fmt:  ## ruff format
	uv run ruff format src tests scripts

lint:  ## ruff check
	uv run ruff check src tests scripts

test:  ## pytest smoke + algorithm suite
	uv run pytest

doctor:  ## environment readiness check
	uv run rl-studio doctor

train:  ## run the numpy GRPO toy loop
	uv run rl-studio train configs/toy-grpo.yaml

eval:  ## evaluate the trained policy vs random baseline
	uv run rl-studio eval digit-sum

sample:  ## sample completions from the trained policy
	uv run rl-studio sample digit-sum --n 5

demo: train eval sample  ## full train -> eval -> sample loop

gpu:  ## explain / launch the scaffolded LLM GRPO path (needs `uv sync --extra gpu`)
	uv run rl-studio gpu-train configs/grpo-qwen.yaml

repro:  ## prove the GRPO toy loop is deterministic (same seed → identical metrics)
	uv run python scripts/check_repro.py -- uv run rl-studio train configs/toy-grpo.yaml

readme:  ## run the README's ci-test commands so the docs can't go stale
	uv run python scripts/test_readme.py

summary:  ## train + print the markdown metrics summary CI posts to the run page
	uv run rl-studio train configs/toy-grpo.yaml --json | uv run python scripts/ci_report.py

clean:
	rm -rf artifacts mlruns mlartifacts mlflow.db metrics.json .pytest_cache .ruff_cache
