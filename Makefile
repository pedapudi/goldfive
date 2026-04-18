ROOT := $(shell pwd)
PROTO_SRC := $(ROOT)/proto
PROTO_OUT := $(ROOT)/goldfive/pb
PROTO_FILES := $(shell find $(PROTO_SRC) -name '*.proto' 2>/dev/null)

.PHONY: help install proto test lint format clean

help:
	@echo "goldfive Makefile targets:"
	@echo "  install   Install package + dev dependencies via uv"
	@echo "  proto     Regenerate Python stubs from proto/ into goldfive/pb/"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff check"
	@echo "  format    Run ruff format"
	@echo "  clean     Remove build, cache, and generated artifacts"

install:
	@cd $(ROOT) && uv sync --extra dev

proto:
	@cd $(ROOT) && uv run --extra proto python -m grpc_tools.protoc \
		--proto_path=$(PROTO_SRC) \
		--python_out=$(PROTO_OUT) \
		--pyi_out=$(PROTO_OUT) \
		$(PROTO_FILES)
	@echo "Proto stubs regenerated into $(PROTO_OUT)"

test:
	@cd $(ROOT) && uv run pytest -q

lint:
	@cd $(ROOT) && uv run ruff check .

format:
	@cd $(ROOT) && uv run ruff format .

clean:
	@rm -rf $(ROOT)/dist $(ROOT)/build $(ROOT)/*.egg-info
	@find $(ROOT) -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find $(ROOT) -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find $(ROOT) -type d -name .ruff_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find $(ROOT) -type d -name .mypy_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean done."
