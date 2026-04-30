# Makefile for mujoco-puppeteer

.PHONY: help setup format test clean run server client server-bg server-stop list run-template demo

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  MJPYTHON := .venv/bin/mjpython
else
  MJPYTHON := .venv/bin/python3
endif

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup    - Set up virtual environment and install dependencies"
	@echo "  make format   - Run YAPF and pre-commit for formatting"
	@echo "  make test     - Run all unit tests"
	@echo "  make run      - Run visual simulation with base template"
	@echo "  make demo     - Run interactive multi-species demo"
	@echo "  make demo-test - Run headless demo test for 10s"
	@echo "  make list      - List available templates"
	@echo "  make run-template name=<template_name> - Run a specific template"
	@echo "  make clean-results - Clean duplicate images and index results"
	@echo "  make kill      - Kill all background simulation processes"
	@echo "  make rerender-all - Re-render all templates as GIFs"
	@echo "  make render template=<path> options=\"<options>\" - Render specific template"
	@echo "  make maintenance - Run full maintenance (tests, leaderboard, renders)"
	@echo "  make clean    - Remove generated files and logs"

# Setup environment
setup:
	@echo "🛠️  Setting up environment..."
	@mkdir -p logs
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv .venv; \
		.venv/bin/pip install -r requirements.txt --upgrade; \
	else \
		echo "Virtual environment already exists."; \
	fi

# Format code
format:

	@echo "🛠️  Running pre-commit validation..."
	@.venv/bin/pre-commit run --all-files
	@echo "✅ All styling and formatting checks passed!"

# Run tests
test: format
	@echo "🧪 Running Unit Tests..."
	@.venv/bin/python3 -m unittest discover -p "*_test.py"

	@echo "✅ All Tests Passed!"

# Run WebSocket server
server:
	@echo "🚀 Starting simulation server..."
	@.venv/bin/python3 server.py

# Run server in background
server-bg:
	@echo "🚀 Starting simulation server in background..."
	@mkdir -p logs
	@.venv/bin/python3 server.py > logs/server.log 2>&1 & echo $$! > logs/server.pid
	@echo "Server started with PID $$(cat logs/server.pid)"

# Stop background server
server-stop:
	@echo "🛑 Stopping simulation server..."
	@if [ -f logs/server.pid ]; then \
		kill $$(cat logs/server.pid) && rm logs/server.pid; \
		echo "Server stopped."; \
	else \
		echo "No PID file found."; \
	fi

# Run WebSocket client
client:
	@echo "🔌 Starting test client..."
	@.venv/bin/python3 client.py

# List available templates
list:
	@.venv/bin/python3 cli.py --list

# Run a specific template
run-template:
	@$(MJPYTHON) cli.py --run $(name)

# Run visual simulation
run:
	@echo "🚀 Running visual simulation..."
	@$(MJPYTHON) cli.py --run base

# Run demo
demo:
	@echo "🎮 Running simulation demo..."
	@$(MJPYTHON) demo.py

# Run demo and record frames
demo-record:
	@echo "🎮 Running simulation demo and recording frames..."
	@.venv/bin/python3 demo.py --record

# Run parallel evolution
parallel-evolve:
	@echo "🚀 Running parallel evolution manager..."
	@.venv/bin/python3 parallel_evolve.py

# Run demo test (headless)
demo-test:
	@echo "🧪 Running demo test in background..."
	@$(MJPYTHON) demo.py --no-viewer & pid=$$! ; sleep 10 ; kill $$pid

# Clean results
clean-results:
	@echo "🧹 Cleaning duplicate images and indexing results..."
	@.venv/bin/python3 maintenance.py

# Run full maintenance
maintenance:
	@echo "🚀 Running full maintenance..."
	@.venv/bin/python3 maintenance.py --tests

# Kill all processes
kill:
	@echo "💀 Killing all simulation and server processes..."
	-pkill -f auto_evolve.py
	-pkill -f cli.py
	-pkill -f demo.py
	-pkill -f server.py
	-pkill -f maintenance.py
	-pkill -f cron_job.py

# Re-render all templates as GIFs
rerender-all:
	@echo "🎬 Re-rendering all templates as GIFs..."
	@PYTHONPATH=. .venv/bin/python3 render.py --batch

# Render specific template as HD image or GIF
render:
	@echo "🎬 Rendering HD image/GIF for $(template)..."
	@.venv/bin/python3 render.py $(template) $(output) $(options)

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -rf logs
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	@echo "Note: .venv was not removed. Remove manually if needed."
