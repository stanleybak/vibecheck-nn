# vibecheck
Vibe-Coded Neural Network Verification Tool - graph branch

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
vibecheck --net model.onnx --spec property.vnnlib
```

## Tests

```bash
pytest tests/ -v
```
