# MountainCar RL — dev commands. Run `make help` to list targets.

.PHONY: help install smoke notebook train train-tabular train-deep tensorboard clean clean-all zip lock requirements

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:        ## Sync dev deps via uv (Python 3.13)
	uv sync

lock:           ## Refresh uv.lock without installing
	uv lock

requirements:   ## Export pinned requirements.txt from uv.lock (for the notebook's !pip install path)
	uv export --no-hashes --format requirements-txt --no-dev > requirements.txt

smoke:          ## ~2 min: train each algo briefly to verify no crashes
	uv run python scripts/smoke.py

notebook:       ## Open JupyterLab on the deliverable folder
	uv run jupyter lab docs/deliverables/task1/

train:          ## ~15-18 min: full matrix, regenerates cached artifacts/
	uv run python -m mountaincar_rl.training.multi_seed --all

train-tabular:  ## ~2 min: tabular agents only
	uv run python -m mountaincar_rl.training.multi_seed --group tabular

train-deep:     ## ~15 min: deep agents only
	uv run python -m mountaincar_rl.training.multi_seed --group deep

tensorboard:    ## Launch TensorBoard on artifacts/tb_logs
	uv run tensorboard --logdir artifacts/tb_logs

clean:          ## Drop tb_logs + checkpoints + figures (keep results JSON cache)
	rm -rf artifacts/tb_logs artifacts/checkpoints artifacts/figures

clean-all:      ## Drop ALL artifacts including the cached results
	rm -rf artifacts/

zip:            ## Build submission zip (usage: GROUP=XX make zip)
	bash scripts/build_submission.sh $(GROUP)
