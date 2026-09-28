"""Rebuild every PDF figure in the current BATA paper from frozen CSVs."""
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
for name in (
    "draw_overview.py",
    "draw_figures_2_4.py",
    "draw_figures_5_6.py",
    "draw_figure5_layout.py",
    "draw_appendix_remaining.py",
):
    subprocess.run([sys.executable, str(root / "tools" / name)], cwd=root, check=True)

# The final paper uses the later 3.3-inch Figure 5 layout.
shutil.copyfile(root / "generated" / "Figure5_Layout.pdf", root / "generated" / "Figure5.pdf")
shutil.copyfile(root / "generated" / "Figure5_Layout.png", root / "generated" / "Figure5.png")

expected = [f"Figure{i}.pdf" for i in range(1, 7)] + [
    "Appendix_Capacity.pdf", "Appendix_ECDF.pdf", "Appendix_G20.pdf",
    "Appendix_Objectives.pdf", "Appendix_Runtime.pdf",
    "Appendix_Sensitivity.pdf", "Appendix_Weights.pdf",
]
missing = [name for name in expected if not (root / "generated" / name).is_file()]
if missing:
    raise SystemExit("Missing paper figures: " + ", ".join(missing))
print(f"Rebuilt {len(expected)} current-paper PDF figures in {root / 'generated'}")
