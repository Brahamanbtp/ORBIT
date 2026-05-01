# Usage: make <target>
# ORBIT helper targets for installation, corpus generation, experiments, testing, cleanup, and help.

.PHONY: install smoke corpus experiment test clean help

install:
	pip install -r requirements.txt

smoke:
	python smoke_test.py

corpus:
	python make_corpus.py

experiment:
	python run_experiments.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf outputs/*
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "Cleaned output and cache files"

help:
	@printf "Available targets:\n"
	@printf "  install     Install Python dependencies from requirements.txt\n"
	@printf "  smoke       Run the ORBIT smoke test\n"
	@printf "  corpus      Generate evaluation corpora\n"
	@printf "  experiment  Run the full ORBIT experiment suite\n"
	@printf "  test        Run the pytest suite\n"
	@printf "  clean       Remove outputs and __pycache__ directories\n"
	@printf "  help        Show this help message\n"