# Makefile for the geometric-fv project

# ====================================================================================
# VARIABLES
# ====================================================================================

# Shell to use
SHELL := /bin/bash

# Project environment
ENV_NAME := envs
ENV_DIR := $(shell pwd)/$(ENV_NAME)

# ====================================================================================
# RUN SCRIPT
# ====================================================================================

.PHONY: run
run:
	@python scripts/burgers.py && feh *.png
	@rm -rf `find . -name '*.png'`

# ====================================================================================
# DEVELOPMENT ENVIRONMENT
# ====================================================================================

.PHONY: env
env: ## Create the conda environment from environment.yaml.
	@if [ -d "$(ENV_DIR)" ]; then \
		echo "Conda environment '$(ENV_NAME)' already exists."; \
		echo "Updating environment '$(ENV_NAME)'..."; \
		conda env update -f environment.yaml --prune; \
	else \
		echo "Creating conda environment '$(ENV_NAME)'..."; \
		conda env create -f environment.yaml; \
	fi

# ====================================================================================
# TESTING, LINTING, AND FORMATTING
# ====================================================================================

# Directories to check
PYTHON_DIRS := src test scripts

.PHONY: test
test: ## Run tests using pytest.
	@echo "Running tests..."
	@pytest --tb=short --quiet

.PHONY: lint
lint:
	@ruff check --output-format=concise $(PYTHON_DIRS)

.PHONY: lint-file
lint-file:
	@ruff check --output-format=concise $(FILE)

.PHONY: format
format:
	@ruff format $(PYTHON_DIRS)

.PHONY: format-file
format-file:
	@ruff format $(FILE)

.PHONY: type
type:
	@echo "Type checking..."
	@mypy $(PYTHON_DIRS)

.PHONY: spell
spell:
	@echo "Spelling..."
	@codespell $(PYTHON_DIRS)


# ====================================================================================
# DOCUMENTATION
# ====================================================================================

.PHONY: docs
docs: ## Build the documentation.
	@echo "Building documentation..."
	@sphinx-build docs/ docs/build/

#
# ====================================================================================
# TAGS
# ====================================================================================

.PHONY: tags
tags:
	@ctags -R --exclude=envs --exclude=docs --exclude=.*/* --exclude=Makefile .


# ====================================================================================
# RESULTS
# ====================================================================================
RESULTS_DIR := results

# Default parameters
EXPERIMENT ?= burgers-highres-full
CFL ?= 2.8
T ?= 0.2

.PHONY: results
results:
	@mkdir -p $(RESULTS_DIR)
	@echo "Running convergence study..."
	@python scripts/convergence.py --experiment $(EXPERIMENT) --dt_dx $(CFL) --t_final $(T)


# ====================================================================================
# CLEANUP
# ====================================================================================

.PHONY: clean
clean: ## Remove temporary files and directories.
	@echo "Cleaning up..."
	@rm -rf `find . -name __pycache__`
	@rm -f `find . -type f -name '*.py[co]'`
	@rm -f `find . -type f -name '*~'`
	@rm -f `find . -type f -name '.*.swp'`
	@rm -rf .pytest_cache
	@rm -rf docs/build
	@rm -rf $(RESULTS_DIR)
