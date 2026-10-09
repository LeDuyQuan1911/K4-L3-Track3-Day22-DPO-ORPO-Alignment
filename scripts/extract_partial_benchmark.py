"""Preserve the real IFEval result printed before the Colab runtime expired."""

import ast
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
notebook = root / "colab" / "Lab22_DPO_Core_Run.ipynb"
cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
output = "".join(
    "".join(item.get("text", []))
    for item in cells[-1]["outputs"]
    if item["output_type"] == "stream"
)
line = next(line for line in output.splitlines() if line.startswith("{'benchmark': 'IFEval'"))
row = ast.literal_eval(line)
assert row["task"] == "ifeval" and row["limit_per_subtask"] == 200
assert row["sft"] == row["dpo"] == 0.51

result = {
    "status": "partial",
    "source": "colab/Lab22_DPO_Core_Run.ipynb, last code cell output",
    "reason": "Colab GPU runtime ended after IFEval; no GSM8K or Global-MMLU-vi score was produced",
    "results": [row],
}
path = root / "data" / "eval" / "benchmark_partial_ifeval.json"
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(path)
