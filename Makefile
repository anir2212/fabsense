PYTHON ?= .venv/bin/python
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export LOKY_MAX_CPU_COUNT := 4
export OPENBLAS_NUM_THREADS := 1

.PHONY: all photo msa utilities figures agent secom test app-test dashboard
all: photo msa utilities figures agent
photo:
	$(PYTHON) -m fabstack.sim_grid
msa:
	$(PYTHON) -m fabstack.msa_run
utilities:
	$(PYTHON) -m fabstack.utilities_run
figures:
	$(PYTHON) -m fabstack.figures
agent:
	$(PYTHON) -m fabstack.agent_eval
secom:
	$(PYTHON) -m fabstack.secom
test:
	$(PYTHON) -m pytest -q
app-test:
	$(PYTHON) -c 'from streamlit.testing.v1 import AppTest; a=AppTest.from_file("dashboards/streamlit_app.py", default_timeout=30).run(); assert not a.exception, [e.message for e in a.exception]; print("Streamlit dashboard passed")'
dashboard:
	.venv/bin/streamlit run dashboards/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
