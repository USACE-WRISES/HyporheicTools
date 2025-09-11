# src/HyporheicTools/__main__.py
# That lets you run: python -m HyporheicTools --yaml examples/basic_case/input/inputs.yaml --figures
from HyporheicTools.cli.main import main
if __name__ == "__main__":
    raise SystemExit(main())