"""JSON / HTML report exporters for analysis results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_json_report(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Strip bulky base64 images from persisted report by default copy
    slim = dict(payload)
    images = slim.get("images")
    if isinstance(images, dict):
        slim["images"] = {
            k: (f"<base64:{len(v)} chars>" if isinstance(v, str) else v)
            for k, v in images.items()
        }
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return out


def render_html_report(payload: dict[str, Any]) -> str:
    """Self-contained HTML report (print-to-PDF friendly)."""
    meta = payload.get("metadata") or {}
    risk = payload.get("risk") or {}
    classification = payload.get("classification") or {}
    tags = payload.get("risk_tags") or []
    rules = (risk.get("rule_hits") or [])
    explain = payload.get("explanation") or {}
    sections = meta.get("sections") or []

    def esc(x: Any) -> str:
        return html.escape(str(x))

    def _safe_b64(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        # Allow only standard base64 alphabet (+ padding / whitespace).
        cleaned = "".join(value.split())
        if not cleaned or len(cleaned) > 8_000_000:
            return None
        import re

        if not re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
            return None
        return cleaned

    section_rows = "\n".join(
        f"<tr><td>{esc(s.get('name'))}</td><td>{esc(s.get('virtual_size'))}</td>"
        f"<td>{esc(s.get('raw_size'))}</td><td>{esc(s.get('entropy'))}</td>"
        f"<td>{'Y' if s.get('is_executable') else ''}</td></tr>"
        for s in sections
    )
    rule_rows = "\n".join(
        f"<tr><td>{esc(r.get('rule_id'))}</td><td>{esc(r.get('severity'))}</td>"
        f"<td>{esc(r.get('weight'))}</td><td>{esc(r.get('detail'))}</td></tr>"
        for r in rules
    )
    feat_rows = "\n".join(
        f"<tr><td>{esc(f.get('feature'))}</td><td>{esc(round(float(f.get('contribution', 0)), 4))}</td></tr>"
        for f in (explain.get("top_features") or [])
    )
    gray = _safe_b64((payload.get("images") or {}).get("grayscale_png_base64"))
    heat = _safe_b64((payload.get("images") or {}).get("heatmap_png_base64"))
    img_html = ""
    if gray:
        img_html += f'<img alt="gray" src="data:image/png;base64,{gray}" />'
    if heat:
        img_html += f'<img alt="heat" src="data:image/png;base64,{heat}" />'

    conf = classification.get("confidence")
    try:
        conf_pct = round(100 * float(conf), 1) if conf is not None else "—"
    except (TypeError, ValueError):
        conf_pct = "—"
    label = classification.get("label") or classification.get("error") or "—"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>bin2img report — {esc(payload.get('filename'))}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#122;}}
h1{{margin:0 0 .25rem}} .muted{{color:#556}}
.score{{font-size:2rem;font-weight:800}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccd;padding:.35rem;text-align:left;font-size:.9rem}}
img{{max-width:48%;image-rendering:pixelated;border:1px solid #333;margin-right:1%}}
.tag{{display:inline-block;border:1px solid #333;padding:.15rem .4rem;margin:.1rem;font-size:.75rem}}
</style></head><body>
<h1>bin2img analysis report</h1>
<p class="muted">{esc(payload.get('filename'))} · {esc(payload.get('analysis_id',''))} · {esc(payload.get('created_at',''))}</p>
<p class="score">{esc(risk.get('verdict','—'))} · risk {esc(risk.get('score','—'))}/100</p>
<p>Model: <b>{esc(label)}</b> ({esc(conf_pct)}%)</p>
<p>{''.join(f'<span class="tag">{esc(t)}</span>' for t in tags)}</p>
<h2>Metadata</h2>
<pre>{esc(json.dumps({k: meta.get(k) for k in ('format','architecture','file_size','entry_point','overall_entropy','high_entropy_alert','is_valid_executable')}, indent=2))}</pre>
<h2>Sections</h2>
<table><thead><tr><th>Name</th><th>Virt</th><th>Raw</th><th>Entropy</th><th>Exec</th></tr></thead>
<tbody>{section_rows or '<tr><td colspan=5>None</td></tr>'}</tbody></table>
<h2>Rule hits</h2>
<table><thead><tr><th>Rule</th><th>Severity</th><th>Weight</th><th>Detail</th></tr></thead>
<tbody>{rule_rows or '<tr><td colspan=4>None</td></tr>'}</tbody></table>
<h2>Top feature contributions</h2>
<table><thead><tr><th>Feature</th><th>Contribution</th></tr></thead>
<tbody>{feat_rows or '<tr><td colspan=2>None</td></tr>'}</tbody></table>
<h2>Visualizations</h2>
<div>{img_html or '<p class="muted">No images</p>'}</div>
<script>/* open and Print → Save as PDF */</script>
</body></html>"""


def write_html_report(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html_report(payload), encoding="utf-8")
    return out
