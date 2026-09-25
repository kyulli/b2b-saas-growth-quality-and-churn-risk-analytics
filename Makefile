.PHONY: notebook notebook-public pipeline report public-comps public-comps-build test

pipeline:
	python -m saas_growth_quality.pipeline --project-root .

report:
	python -m saas_growth_quality.pipeline --project-root . --stages report

public-comps:
	python scripts/fetch_sec_companyfacts.py --project-root .
	python scripts/build_public_comps_panel.py --project-root .

public-comps-build:
	python scripts/build_public_comps_panel.py --project-root .

notebook:
	jupyter notebook notebooks/saas_growth_quality_churn_risk.ipynb

notebook-public:
	jupyter notebook notebooks/02_public_saas_business_model_evolution.ipynb

test:
	pytest -q
