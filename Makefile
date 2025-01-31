# ------------------------------------
# Customized func / var define
# ------------------------------------

HAS_POETRY := $(shell command -v poetry 2> /dev/null)
POETRY_VERSION := $(shell poetry version $(shell git describe --tags --abbrev=0))

# ------------------------------------
# Test
# ------------------------------------

.PHONY: unittests
## Run unittests
unittests:
	poetry run python -m pytest

.PHONY: clean_tests
## Remove pytest cache and junit report after tests
clean_tests:
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type f -name '*junit_report.xml' -exec rm {} +

.PHONY: coverage
## Combine and build final coverage
coverage:
	coverage run -m pytest
	coverage combine --data-file .coverage || true
	coverage html -i
	coverage report -i

# ------------------------------------
# Build package
# ------------------------------------

.PHONY: build_pkg
## Build the package using poetry
build_pkg:
	@echo "Start to build pkg"
ifdef HAS_POETRY
	@$(POETRY_VERSION)
	poetry build
else
	@echo "To build the package, you need to have poetry first"
	exit 1
endif

.PHONY: build
## Clean up cache from previous built, and build the package
build: clean_built build_pkg

.PHONY: clean_built
## Remove cache, built package, and docs directories after build or installation
clean_built:
	find . -type d -name dist -exec rm -r {} +
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# ------------------------------------
# Build doc
# ------------------------------------

.PHONY: docs_deploy
## Build docs using mike
docs_deploy:
	@echo "Starting to build docs"
	@echo "more info: https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/"
ifdef HAS_POETRY
	@$(POETRY_VERSION)
	poetry version -s | xargs -I {} sh -c 'echo Deploying version {} && mike deploy --push --update-aliases {} latest'
else
	@echo "To build the docs, you need to have poetry first"
	exit 1
endif

.PHONY: docs_version_list
## List available versions of the docs
docs_version_list:
	mike list

.PHONY: docs_version_serve
## Serve versioned docs
docs_version_serve:
	@echo "Start to serve versioned docs"
	mike serve

.PHONY: docs
## Create or Deploy MkDocs based documentation to GitHub pages.
deploy_doc:
	mkdocs gh-deploy

.PHONY: serve_doc
## Test MkDocs based documentation locally.
serve_doc:
	poetry run mkdocs serve

# ------------------------------------
# Clean All
# ------------------------------------

.PHONY: clean
## Remove cache, built package, and docs directories after build or installation
clean:
	find . -type d -name dist -exec rm -r {} +
	find . -type f -name '*.rst' ! -name 'index.rst' -delete
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# ------------------------------------
# Default
# ------------------------------------

.DEFAULT_GOAL := help

help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
