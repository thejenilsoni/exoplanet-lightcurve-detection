.PHONY: install install-api install-web dev dev-api dev-web test lint build

install: install-api install-web

install-api:
	python3 -m venv .venv
	.venv/bin/pip install -e 'apps/api[dev,ml]'

install-web:
	npm --prefix apps/web install

dev:
	$(MAKE) -j2 dev-api dev-web

dev-api:
	.venv/bin/uvicorn app.main:app --app-dir apps/api --reload --port 8000

dev-web:
	npm --prefix apps/web run dev

test:
	PYTHONPATH=apps/api .venv/bin/pytest apps/api/tests
	npm --prefix apps/web run typecheck

lint:
	.venv/bin/ruff check apps/api ml scripts
	npm --prefix apps/web run lint

build:
	npm --prefix apps/web run build
