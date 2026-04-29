# Makefile for mujoco-puppeteer

.PHONY: help setup format test clean run server client server-bg server-stop list run-template

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup    - Set up virtual environment and install dependencies"
	@echo "  make format   - Run YAPF and pre-commit for formatting"
	@echo "  make test     - Run all unit tests"
	@echo "  make run      - Run visual simulation"
	@echo "  make server    - Run WebSocket simulation server"
	@echo "  make server-bg - Run server in background"
	@echo "  make server-stop - Stop background server"
	@echo "  make client    - Run WebSocket test client"
	@echo "  make list      - List available templates"
	@echo "  make run-template name=<template_name> - Run a specific template"
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
	@echo "🖌️  Forcing YAPF Python Formatting..."
	@git ls-files '*.py' | xargs .venv/bin/yapf -i --style="{based_on_style: google, indent_width: 2, column_limit: 80}"
	@echo "🛠️  Running pre-commit validation..."
	@.venv/bin/pre-commit run --all-files
	@echo "✅ All styling and formatting checks passed!"

# Run tests
test: format
	@echo "🧪 Running All Tests..."
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
	@.venv/bin/mjpython cli.py --run $(name)

# Run visual simulation
run:
	@echo "🚀 Running visual simulation..."
	@.venv/bin/mjpython simulate_visual.py

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -rf logs
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	@echo "Note: .venv was not removed. Remove manually if needed."
