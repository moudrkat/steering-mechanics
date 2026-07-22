.PHONY: install demo test clean

install:          ## pip install the package (editable) with deps
	pip install -e .

demo:             ## render figures from shipped example data — NO GPU needed
	python -m steermech.plot

test:             ## unit tests for the scoring logic (no server needed)
	python -m pytest tests_steermech.py -q

clean:
	rm -rf fig/dose_response.png fig/component_attribution.png __pycache__ steermech/__pycache__
