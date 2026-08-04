.DEFAULT_GOAL := help

.PHONY: help install test lint format check build

help: ## 사용 가능한 명령 표시
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## 개발 의존성 설치
	uv sync --extra dev

test: ## 테스트 실행
	uv run pytest

lint: ## 정적 검사 실행
	uv run ruff check .

format: ## 코드 포맷 적용
	uv run ruff format .

check: ## 포맷, 린트, 테스트 검증
	uv run ruff format --check .
	uv run ruff check .
	uv run pytest

build: check ## wheel과 source distribution 생성
	uv build
