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

.PHONY: test
test: ## Run tests using pytest.
	@echo "Running tests..."
	@pytest

.PHONY: lint
lint: ## Run linters and formatters using pre-commit on all files.
	@echo "Running pre-commit hooks for linting and formatting..."
	@pre-commit run --all-files

.PHONY: format
format: ## Run linters and formatters using pre-commit on all files.
	@echo "Running pre-commit hooks for linting and formatting..."
	@pre-commit run --all-files


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
