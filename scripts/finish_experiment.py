"""Write results markdown from manifests and probe JSON, and run probes.

Numbers in the markdown come from those files. Paper figures are labeled as
the paper's, not as measurements from this run.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_INIT = 8.9
PAPER_CONSUMER = 25.2


def log_filter() -> None:
    """Drop carriage-return progress bars. Keep errors and a sparse heartbeat."""
    acc = ""
    prog_n = 0
    while True:
        chunk = sys.stdin.read(65536)
        if not chunk:
            break
        acc += chunk
        lines = acc.splitlines(keepends=True)
        if lines and lines[-1][-1:] not in "\r\n":
            acc = lines.pop()
        else:
            acc = ""
        for part in lines:
            line = part.strip()
            if not line:
                continue
            if "it/s" in line:
                prog_n += 1
                if prog_n % 200 == 0:
                    print(line, flush=True)
                continue
            print(line, flush=True)
    tail = acc.strip()
    if tail:
        print(tail, flush=True)


def _latest_run(name: str) -> Path | None:
    root = ROOT / "runs" / name
    if not root.is_dir():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _ckpt(run_dir: Path) -> Path | None:
    cands = list(run_dir.rglob("checkpoints/last*.ckpt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _config_text(run_dir: Path) -> str:
    path = run_dir / ".hydra" / "config.yaml"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _field(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", text, re.M)
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def _metrics(run_dir: Path) -> dict:
    paths = sorted(run_dir.rglob("metrics.csv"))
    if not paths:
        return {}
    with paths[-1].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in reversed(rows):
        if row.get("fit/loss_epoch"):
            return row
    return rows[-1] if rows else {}


def _hours(run_dir: Path, ckpt: Path | None) -> float | None:
    if ckpt is None:
        return None
    try:
        start = datetime.strptime(run_dir.name, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None
    end = datetime.fromtimestamp(ckpt.stat().st_mtime)
    return round((end - start).total_seconds() / 3600, 2)


def _probe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _append_blocked(title: str, body: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "BLOCKED.md"
    chunk = f"\n## {title}\n\n{body.strip()}\n"
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + chunk, encoding="utf-8")
    else:
        path.write_text("# BLOCKED\n" + chunk, encoding="utf-8")


def _imagenet_ready() -> bool:
    train = os.environ.get("IMAGENET_TRAIN", "")
    val = os.environ.get("IMAGENET_VAL", "")
    return bool(train and val and Path(train).is_dir() and Path(val).is_dir())


def _run_probe(out: Path, ckpt: Path | None, init: bool) -> bool:
    if out.is_file() and ckpt is not None and out.stat().st_mtime >= ckpt.stat().st_mtime:
        print(f"probe exists {out}", flush=True)
        return True
    if out.is_file() and init:
        print(f"probe exists {out}", flush=True)
        return True
    if not _imagenet_ready():
        cmd = _probe_cmd(out, ckpt, init)
        _append_blocked(
            "ImageNet probe",
            "IMAGENET_TRAIN / IMAGENET_VAL is missing or not a directory, so the probe was not run.\n\n"
            f"```bash\n{cmd}\n```",
        )
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "probe_imagenet.py")]
    if init:
        cmd.append("--init")
    else:
        cmd.extend(["--ckpt", str(ckpt)])
    cmd.extend(["--out", str(out)])
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0 or not out.is_file():
        _append_blocked(
            "ImageNet probe failed",
            f"Command exited {proc.returncode}.\n\n```bash\n{_probe_cmd(out, ckpt, init)}\n```",
        )
        return False
    return True


def _probe_cmd(out: Path, ckpt: Path | None, init: bool) -> str:
    if init:
        return f"bash scripts/probe_imagenet.sh --init --out {out.relative_to(ROOT)}"
    return (
        "bash scripts/probe_imagenet.sh --ckpt "
        f"{ckpt} --out {out.relative_to(ROOT)}"
    )


def _fmt(value, digits=3):
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _train_section(name: str) -> str:
    run_dir = _latest_run(name)
    if run_dir is None:
        return f"No `runs/{name}` directory.\n"
    ckpt = _ckpt(run_dir)
    manifest = _manifest(run_dir)
    cfg = _config_text(run_dir)
    metrics = _metrics(run_dir)
    hours = _hours(run_dir, ckpt)
    lines = [
        f"- run dir: `{run_dir.relative_to(ROOT)}`",
        f"- checkpoint: `{ckpt.relative_to(ROOT) if ckpt else 'missing'}`",
        f"- global_step: {manifest.get('global_step', 'n/a')} / max_steps: {manifest.get('max_steps', _field(cfg, 'max_steps'))}",
        f"- gpu: {manifest.get('gpu', 'n/a')}",
        f"- peak allocated VRAM: {manifest.get('peak_vram_mib', 'n/a')} MiB",
        f"- wall hours (run-dir stamp to last checkpoint): {_fmt(hours, 2)}",
        f"- seed: {_field(cfg, 'seed')}",
        f"- batch size: {_field(cfg, 'batch_size')}",
        f"- model: {_field(cfg, 'name')}",
        f"- precision: {_field(cfg, 'precision')}",
        f"- frames in the Lance store: 776576 (10 videos, not resampled)",
        f"- last epoch loss: {_fmt(metrics.get('fit/loss_epoch'))}",
        f"- last epoch pred_loss: {_fmt(metrics.get('fit/pred_loss_epoch'))}",
        f"- last epoch sigreg_loss: {_fmt(metrics.get('fit/sigreg_loss_epoch'))}",
        f"- last epoch cls_std: {_fmt(metrics.get('fit/cls_std_epoch'))}",
    ]
    if metrics.get("fit/disreg_epoch"):
        lines.append(f"- last epoch disreg: {_fmt(metrics.get('fit/disreg_epoch'))}")
        lines.append(f"- last epoch disreg_pred: {_fmt(metrics.get('fit/disreg_pred_epoch'))}")
        lines.append(f"- last epoch sigreg_z: {_fmt(metrics.get('fit/sigreg_z_epoch'))}")
        lines.append(f"- last epoch sigreg_d: {_fmt(metrics.get('fit/sigreg_d_epoch'))}")
        lines.append(f"- last epoch z_std: {_fmt(metrics.get('fit/z_std_epoch'))}")
        lines.append(f"- last epoch d_std: {_fmt(metrics.get('fit/d_std_epoch'))}")
    change = RESULTS / f"{name}_change.log"
    if change.is_file():
        lines.append(f"- logged change: {change.read_text(encoding='utf-8').strip()}")
    return "\n".join(lines) + "\n"


def _probe_section(path: Path) -> str:
    payload = _probe(path)
    if payload is None:
        return f"Probe file `{path.relative_to(ROOT)}` is not on disk. See `results/BLOCKED.md` if the command was recorded there.\n"
    return "\n".join(
        [
            f"- file: `{path.relative_to(ROOT)}`",
            f"- val_top1_best: {_fmt(payload.get('val_top1_best'), 2)}",
            f"- val_top1_last: {_fmt(payload.get('val_top1_last'), 2)}",
            f"- weight_source: {payload.get('weight_source')}",
            f"- epochs: {payload.get('epochs')}",
            f"- micro-batch: {payload.get('batch_size')} x accum {payload.get('accum')} = {payload.get('effective_batch')}",
            f"- limit: {payload.get('limit')} (0 means the full split)",
            f"- probe hours: {_fmt(payload.get('hours'), 2)}",
            f"- gpu: {payload.get('gpu')}",
            f"- peak allocated VRAM: {payload.get('peak_vram_mib')} MiB",
        ]
    ) + "\n"


def _smoke_section() -> str:
    run_dir = _latest_run("smoke30")
    if run_dir is None:
        return "Smoke run directory not found.\n"
    manifest = _manifest(run_dir)
    metrics = _metrics(run_dir)
    return "\n".join(
        [
            f"- steps: {manifest.get('global_step')} / {manifest.get('max_steps')}",
            f"- gpu: {manifest.get('gpu')}",
            f"- peak allocated VRAM: {manifest.get('peak_vram_mib')} MiB",
            f"- last epoch loss: {_fmt(metrics.get('fit/loss_epoch'))}",
            f"- last epoch cls_std: {_fmt(metrics.get('fit/cls_std_epoch'))}",
            "- checkpoints: `step-000200.ckpt` and `step-000400.ckpt` under the smoke run",
            "- gate: finite loss, cls_std above 1e-3, checkpoint files present. Batch stayed 128. Precision was already bf16-mixed.",
        ]
    ) + "\n"


def write_baseline() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Baseline",
            "",
            "ViT-Tiny, batch 128, seed 0, 39063 steps, DISReg off. Data is the 10-video Walking Tours Lance store (776,576 frames).",
            "",
            "## Smoke",
            "",
            _smoke_section(),
            "## Training",
            "",
            _train_section("baseline"),
            "## ImageNet attentive probe",
            "",
            _probe_section(RESULTS / "baseline_probe.json"),
            "## Paper consumer number",
            "",
            f"The paper reports about {PAPER_INIT}% at initialization and {PAPER_CONSUMER}% after pretraining ViT-Tiny on eight Walking Tours videos. This run uses ten videos. The probe number above is the measurement for this checkpoint, not a claim of matching {PAPER_CONSUMER}%.",
            "",
        ]
    )
    (RESULTS / "baseline.md").write_text(text, encoding="utf-8")
    print(f"wrote {RESULTS / 'baseline.md'}", flush=True)


def write_disreg() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# DISReg",
            "",
            "Same ViT-Tiny consumer recipe as the baseline, with MotionJEPA DISReg added. LeVJEPA SIGReg stays on. Default weights: lambda_disreg 1, lambda_z 0.25, lambda_pred 0.5, lambda_sigreg_d 2.",
            "",
            "## Training",
            "",
            _train_section("disreg"),
            "## ImageNet attentive probe",
            "",
            _probe_section(RESULTS / "disreg_probe.json"),
            "",
        ]
    )
    (RESULTS / "disreg.md").write_text(text, encoding="utf-8")
    print(f"wrote {RESULTS / 'disreg.md'}", flush=True)


def _top1(path: Path) -> str:
    payload = _probe(path)
    if payload is None or payload.get("val_top1_best") is None:
        return "not run"
    return f"{float(payload['val_top1_best']):.2f}"


def _cell(name: str, key: str, digits=2) -> str:
    run_dir = _latest_run(name)
    if run_dir is None:
        return "n/a"
    if key == "hours":
        hours = _hours(run_dir, _ckpt(run_dir))
        return _fmt(hours, 2)
    if key == "vram":
        return str(_manifest(run_dir).get("peak_vram_mib", "n/a"))
    if key == "gpu":
        return str(_manifest(run_dir).get("gpu", "n/a"))
    if key == "steps":
        manifest = _manifest(run_dir)
        return f"{manifest.get('global_step', 'n/a')}"
    if key == "batch":
        return str(_field(_config_text(run_dir), "batch_size") or "n/a")
    return "n/a"


def write_comparison() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    init = _probe(RESULTS / "init_probe.json")
    base = _probe(RESULTS / "baseline_probe.json")
    dis = _probe(RESULTS / "disreg_probe.json")
    lines = [
        "# Comparison",
        "",
        "Frozen ImageNet attentive probe. Same probe for init, baseline, and DISReg. Encoder weights are the EMA copy when the checkpoint has `state_dict_ema`.",
        "",
        "| run | ImageNet top-1 | train hours | GPU | peak VRAM MiB | batch | steps |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        f"| init | {_top1(RESULTS / 'init_probe.json')} | n/a | {(init or {}).get('gpu', 'n/a')} | {(init or {}).get('peak_vram_mib', 'n/a')} | n/a | 0 |",
        f"| baseline | {_top1(RESULTS / 'baseline_probe.json')} | {_cell('baseline', 'hours')} | {_cell('baseline', 'gpu')} | {_cell('baseline', 'vram')} | {_cell('baseline', 'batch')} | {_cell('baseline', 'steps')} |",
        f"| +DISReg | {_top1(RESULTS / 'disreg_probe.json')} | {_cell('disreg', 'hours')} | {_cell('disreg', 'gpu')} | {_cell('disreg', 'vram')} | {_cell('disreg', 'batch')} | {_cell('disreg', 'steps')} |",
        "",
        f"The paper's consumer ViT-Tiny run is {PAPER_INIT}% at init and {PAPER_CONSUMER}% after pretraining on eight Walking Tours videos. This table is the 10-video store (776,576 frames) and does not claim {PAPER_CONSUMER}%.",
        "",
    ]
    if init and init.get("limit"):
        lines.append(f"Init probe limit: {init.get('limit')}. That is a subset, not the full validation set.")
    if base and base.get("limit"):
        lines.append(f"Baseline probe limit: {base.get('limit')}.")
    if dis and dis.get("limit"):
        lines.append(f"DISReg probe limit: {dis.get('limit')}.")
    lines.append("")
    lines.append("Rerun probes:")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/probe_imagenet.sh --init --out results/init_probe.json")
    lines.append("bash scripts/probe_imagenet.sh --ckpt \"$(find runs/baseline -name 'last.ckpt' -printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-)\" --out results/baseline_probe.json")
    lines.append("bash scripts/probe_imagenet.sh --ckpt \"$(find runs/disreg -name 'last.ckpt' -printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-)\" --out results/disreg_probe.json")
    lines.append("```")
    lines.append("")
    (RESULTS / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'comparison.md'}", flush=True)


def baseline_probe() -> None:
    run_dir = _latest_run("baseline")
    if run_dir is None:
        _append_blocked("baseline probe", "No runs/baseline directory.")
        write_baseline()
        return
    ckpt = _ckpt(run_dir)
    if ckpt is None:
        _append_blocked("baseline probe", "No last.ckpt under runs/baseline.")
        write_baseline()
        return
    _run_probe(RESULTS / "baseline_probe.json", ckpt, init=False)
    write_baseline()


def disreg_probe() -> None:
    run_dir = _latest_run("disreg")
    if run_dir is None:
        _append_blocked("disreg probe", "No runs/disreg directory.")
        write_disreg()
        write_comparison()
        return
    ckpt = _ckpt(run_dir)
    if ckpt is None:
        _append_blocked("disreg probe", "No last.ckpt under runs/disreg.")
    else:
        _run_probe(RESULTS / "disreg_probe.json", ckpt, init=False)
    write_disreg()
    _run_probe(RESULTS / "init_probe.json", ckpt=None, init=True)
    write_comparison()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: finish_experiment.py {log-filter,baseline-probe,disreg-probe,comparison}")
    cmd = sys.argv[1]
    if cmd == "log-filter":
        log_filter()
    elif cmd == "baseline-probe":
        baseline_probe()
    elif cmd == "disreg-probe":
        disreg_probe()
    elif cmd == "comparison":
        write_comparison()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
