.PHONY: all test experiments figures clean

PY := python3

all: test experiments figures

## Run the test suite (124 tests)
test:
	$(PY) -m pytest

## Measure everything -> reports/*.csv
experiments:
	$(PY) scripts/run_experiments.py

## Render the figures -> reports/figures/*.png
figures:
	$(PY) scripts/make_figures.py

clean:
	rm -rf reports/*.csv reports/*.json reports/figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
