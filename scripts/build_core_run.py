"""Make a fresh NB0–NB4 Colab template without overwriting executed outputs."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent.parent
source = root / "colab" / "Lab22_DPO_T4.ipynb"
target = root / "colab" / "Lab22_DPO_Core_Template.ipynb"
notebook = json.loads(source.read_text(encoding="utf-8"))
cells = notebook["cells"]
stop = next(i for i, cell in enumerate(cells) if "notebooks/05_merge_deploy_gguf.py" in "".join(cell["source"]))
bonus_start = next(i for i, cell in enumerate(cells) if "notebooks/03b_dpo_variants.py" in "".join(cell["source"]))
core_resume = next(i for i, cell in enumerate(cells) if "notebooks/04_compare_and_eval.py" in "".join(cell["source"]))
notebook["cells"] = cells[:bonus_start] + cells[core_resume:stop]

install = notebook["cells"][3]
install["source"] = [
    line.replace(
        ' "llama-cpp-python>=0.3.16,<1.0" "lm-eval[ifeval,math]>=0.4.13,<0.5"',
        '',
    ).replace(' "openai>=1.55,<4.0" "anthropic>=0.40,<2.0"', '')
    for line in install["source"]
]

export = r'''from pathlib import Path
import zipfile
from google.colab import files

root = Path('/content/lab22')
archive = Path('/content/lab22-core-evidence.zip')
patterns = (
    'submission/screenshots/*.png', 'data/pref/*.parquet', 'data/pref/*.json',
    'data/eval/*.json', 'data/eval/*.jsonl',
    'adapters/sft-mini/*.json', 'adapters/dpo/*.json',
    'models/sft-merged/config.json',
)
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for pattern in patterns:
        for path in root.glob(pattern):
            z.write(path, path.relative_to(root))
print(f'Exported {archive.stat().st_size / 1e6:.1f} MB to {archive}')
files.download(str(archive))
'''
notebook["cells"].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": export.splitlines(keepends=True),
})
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {target} ({len(notebook['cells'])} cells)")
