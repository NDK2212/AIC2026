SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
FRONTEND_DIR := src/web/frontend

BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 7860
BACKEND_URL ?= http://127.0.0.1:$(BACKEND_PORT)
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173
CONFIG ?= config/config.yaml

.PHONY: help install install-backend install-frontend backend frontend full build-frontend test

help:
	@echo "AIC 2026 development commands"
	@echo ""
	@echo "  make install    Create .venv and install Python + frontend dependencies"
	@echo "  make backend    Run FastAPI/Uvicorn on http://localhost:$(BACKEND_PORT)"
	@echo "  make frontend   Run Vite on http://localhost:$(FRONTEND_PORT)"
	@echo "  make full       Run backend and frontend together"
	@echo "  make build-frontend  Build the React bundle served by FastAPI"
	@echo "  make test       Run the Python test suite"

install: install-backend install-frontend
	@if [[ -f .env ]]; then \
		echo "Using existing $(CURDIR)/.env"; \
	else \
		cp .env.example .env; \
		echo "Created $(CURDIR)/.env from .env.example; fill in the API keys before running."; \
	fi
	@echo "Installation complete. Run 'make full' or 'make backend'."

$(VENV_PYTHON):
	$(PYTHON) -m venv "$(VENV)"

install-backend: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

install-frontend:
	@command -v npm >/dev/null 2>&1 || { echo "npm is required (Node.js 20+ recommended)."; exit 1; }
	npm --prefix "$(FRONTEND_DIR)" ci

backend:
	@test -x "$(VENV_PYTHON)" || { echo "Missing $(VENV_PYTHON). Run 'make install' first."; exit 1; }
	$(VENV_PYTHON) -m src.cli ui --config "$(CONFIG)" --host "$(BACKEND_HOST)" --port "$(BACKEND_PORT)"

frontend:
	@test -d "$(FRONTEND_DIR)/node_modules" || { echo "Frontend dependencies are missing. Run 'make install' first."; exit 1; }
	VITE_API_TARGET="$(BACKEND_URL)" npm --prefix "$(FRONTEND_DIR)" run dev -- --host "$(FRONTEND_HOST)" --port "$(FRONTEND_PORT)"

full:
	@set -m; \
	$(MAKE) --no-print-directory backend & backend_pid=$$!; \
	$(MAKE) --no-print-directory frontend & frontend_pid=$$!; \
	cleanup() { \
		kill $$backend_pid $$frontend_pid 2>/dev/null || true; \
		wait $$backend_pid $$frontend_pid 2>/dev/null || true; \
	}; \
	trap cleanup INT TERM EXIT; \
	status=0; \
	wait -n $$backend_pid $$frontend_pid || status=$$?; \
	cleanup; \
	trap - INT TERM EXIT; \
	exit $$status

build-frontend:
	@test -d "$(FRONTEND_DIR)/node_modules" || { echo "Frontend dependencies are missing. Run 'make install' first."; exit 1; }
	npm --prefix "$(FRONTEND_DIR)" run build

test:
	@test -x "$(VENV_PYTHON)" || { echo "Missing $(VENV_PYTHON). Run 'make install' first."; exit 1; }
	$(VENV_PYTHON) -m pytest -q
