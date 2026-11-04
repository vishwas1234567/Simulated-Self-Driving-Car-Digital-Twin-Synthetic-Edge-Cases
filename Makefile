# Full pipeline (about 15 min on 2 CPU cores). Run from the project root.
PY ?= python3
.PHONY: all test results figures pdf clean
all: results figures pdf
test:
	cd code && $(PY) tests.py
results:
	cd code && $(PY) run_experiments.py > ../logs/experiments.log
	cd code && $(PY) run_twin.py 5000 > ../logs/twin.log
	cd code && $(PY) run_stack.py > ../logs/stack.log
	cd code && $(PY) embedded.py > ../logs/embedded.log
figures:
	cd code && $(PY) make_figures.py && $(PY) make_figures_stack.py
pdf:
	latexmk -pdf -interaction=nonstopmode main.tex
clean:
	rm -rf build/* && latexmk -C main.tex
