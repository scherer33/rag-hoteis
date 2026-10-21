.PHONY: build up down logs test lint fmt index eval shell

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

test:
	docker compose run --rm dev pytest

lint:
	docker compose run --rm dev ruff check .

fmt:
	docker compose run --rm dev ruff format .

index:
	docker compose run --rm dev python scripts/build_index.py --corpus data/corpus --qdrant-url http://qdrant:6333

eval:
	docker compose run --rm dev python -m hotel_rag.eval.run

shell:
	docker compose run --rm dev bash
