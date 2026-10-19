SHELL = /bin/bash
VENV_NAME = tsfb_env
override BOOTSTRAP_PYTHON := python3.11
override PYTHON := ./$(VENV_NAME)/bin/python3.11
MAIN_FOLDER = tsfb
TEST_FOLDER = tests

IGNORE_FOLDERS := tsfb/base/origin_code

IGNORE_BLACK := $(subst $(space),|,$(IGNORE_FOLDERS))
IGNORE_FLAKE8 := $(subst $(space),$(comma),$(IGNORE_FOLDERS))
IGNORE_ISORT := $(IGNORE_FOLDERS)
IGNORE_MYPY := $(IGNORE_BLACK)
IGNORE_PYTEST := $(foreach f,$(IGNORE_FOLDERS),--ignore=$(f))
IGNORE_PYLINT := $(shell find ${MAIN_FOLDER} -name "*.py" $(foreach d,$(IGNORE_FOLDERS), -not -path "$(d)/*"))

.PHONY: venv style test

# Environment
venv:
	${BOOTSTRAP_PYTHON} -m venv ${VENV_NAME} && \
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install -e .[dev] && \
	${PYTHON} -m pre_commit install

# Style
style:
	@echo "--- Ignoring: $(IGNORE_FOLDERS) ---"
	${PYTHON} -m black ./${MAIN_FOLDER}/ --exclude '($(IGNORE_BLACK))'
	${PYTHON} -m flake8 ./${MAIN_FOLDER}/ --exclude=$(IGNORE_FLAKE8)
	${PYTHON} -m isort ./${MAIN_FOLDER}/ --skip $(IGNORE_ISORT)

test:
	${PYTHON} -m flake8 ./${MAIN_FOLDER}/ --exclude=$(IGNORE_FLAKE8)
	${PYTHON} -m mypy $(foreach d,$(IGNORE_FOLDERS),--exclude $(d)) ./${MAIN_FOLDER}/
	CUDA_VISIBLE_DEVICES=""  ${PYTHON} -m pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/ $(IGNORE_PYTEST)
	${PYTHON} -m pylint $(IGNORE_PYLINT)
