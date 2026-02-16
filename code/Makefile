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
	@pytest

.PHONY: ruff-check
ruff-check:
	@echo "Running ruff for linting..."
	@ruff check $(PYTHON_DIRS)

.PHONY: ruff-format
ruff-format:
	@echo "Running ruff for formatting..."
	@ruff format $(PYTHON_DIRS)

.PHONY: mypy
mypy:
	@echo "Running mypy for type checking..."
	@mypy $(PYTHON_DIRS)

.PHONY: codespell
codespell:
	@echo "Running codespell for spelling..."
	@codespell $(PYTHON_DIRS)


# ====================================================================================
# DOCUMENTATION
# ====================================================================================

.PHONY: docs
docs: ## Build the documentation.
	@echo "Building documentation..."
	@sphinx-build docs/ docs/build/


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
