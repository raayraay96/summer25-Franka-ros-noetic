SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= python3
IMAGE ?= franka-teleop:noetic
CI_RESULTS ?= results/v1.1-local

.PHONY: help install test lint validate ci docker-build sim sim-compose qp-smoke qp-audit benchmark-smoke benchmark-full full-validation telemetry-test clean

help: ## Show the available developer commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install pinned Python dependencies
	$(PYTHON) -m pip install -r requirements-ci.txt

test: ## Run pure-Python unit and property tests
	$(PYTHON) -m pytest -q tests/

telemetry-test: ## Run only telemetry queue and serialization tests
	$(PYTHON) -m pytest -q tests/test_telemetry.py

lint: ## Check formatting and static style
	$(PYTHON) -m black --check src/vision_arm_control/src tests benchmarks scripts ros2_ws/src/franka_teleop_ros2
	$(PYTHON) -m flake8 src/vision_arm_control/src tests benchmarks scripts ros2_ws/src/franka_teleop_ros2

validate: ## Validate configs, docs, launch XML, and repository hygiene
	$(PYTHON) scripts/validate_repository.py
	$(PYTHON) scripts/validate_config.py

qp-smoke: ## Run the CI-sized QP oracle cross-validation
	mkdir -p $(CI_RESULTS)
	$(PYTHON) scripts/qp_cross_validation.py --generic 400 --cbf 400 --output $(CI_RESULTS)/qp-smoke.json

qp-audit: ## Run the full 6,000-seed QP correctness audit
	$(PYTHON) scripts/qp_cross_validation.py

benchmark-smoke: ## Run the lightweight v1.1 benchmark subset
	mkdir -p $(CI_RESULTS)
	$(PYTHON) benchmarks/benchmark_retargeting_v11.py --smoke --output $(CI_RESULTS)

benchmark-full: ## Run the 30-replicate paired simulation benchmark
	$(PYTHON) benchmarks/benchmark_paired_v11.py --replicates 30

docker-build: ## Build the reproducible ROS Noetic container
	docker build -t $(IMAGE) .

sim: docker-build ## Run the headless ROS Noetic simulation smoke path
	docker run --rm $(IMAGE) bash scripts/ci_smoke_noetic.sh

sim-compose: ## Run the same simulation path through Docker Compose
	docker compose run --rm noetic-sim

ci: lint test validate qp-smoke benchmark-smoke ## Reproduce the pull-request quality gate locally

full-validation: test validate qp-audit benchmark-full ## Run the full local validation evidence path

clean: ## Remove local transient validation output
	rm -rf $(CI_RESULTS) .pytest_cache
