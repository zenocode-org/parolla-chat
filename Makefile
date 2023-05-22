.PHONY: start
start:
	uvicorn main:app --reload --port 9000 --log-config log.ini

.PHONY: format
format:
	black .
	isort .