PY ?= python
.PHONY: install install-dev data train bench bench-quick sample test ui clean

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev,ui]"

data:            ## download + cache CIFAR-10 (torchvision, HF parquet fallback)
	$(PY) -c "from studio.data import load_cifar10; load_cifar10('data', True); load_cifar10('data', False)"

train:           ## train vanilla + PGD-AT CNN and a vanilla ResNet (skips existing checkpoints)
	$(PY) experiments/train_models.py --config config/default.yaml

bench: train     ## full benchmark -> results/report.md
	$(PY) experiments/run_benchmark.py --config config/default.yaml --out results

bench-quick:     ## tiny smoke run (needs checkpoints)
	$(PY) experiments/run_benchmark.py --quick --out results/quick

sample:          ## refresh the committed sample report from the latest bench
	rm -rf results/sample && mkdir -p results/sample && cp -r results/report.md results/report.html results/results.json results/figures results/sample/

test:
	$(PY) -m pytest -q

ui:
	streamlit run app/ui.py

clean:
	rm -rf results/quick .pytest_cache **/__pycache__
