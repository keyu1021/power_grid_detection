.PHONY: help label-help label-init label-batch label-list label-progress label-status clean

help:
	@echo "Power Grid Detection - Available commands"
	@echo ""
	@echo "Labeling Commands:"
	@echo "  make label-help       Show labeling workflow help"
	@echo "  make label-init       Initialize QGIS labeling setup"
	@echo "  make label-batch N=5  Create QGIS projects for N tiles (default: 5)"
	@echo "  make label-list       List all tiles and their status"
	@echo "  make label-progress   Show labeling progress"
	@echo "  make label-status ST=<status>  Filter tiles by status (todo/in_progress/labeled/reviewed)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Remove generated files"

label-help:
	@cat src/labeling/QGIS_WORKFLOW.md

label-init:
	python src/labeling/qgis_label.py init

label-batch:
	@N=$${N:-5}; echo "Creating QGIS projects for $$N tiles..."; \
	python src/labeling/qgis_label.py create-batch $$N

label-list:
	python src/labeling/qgis_label.py list

label-progress:
	python src/labeling/qgis_label.py progress

label-status:
	@ST=$${ST:-todo}; echo "Tiles with status: $$ST"; \
	python src/labeling/qgis_label.py list --status $$ST

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .venv -prune -o -type d -name venv -prune
	@echo "Cleanup complete"
