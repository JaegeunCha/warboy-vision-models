#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Warboy Outputs Viewer (+ latest performance_result metrics table)

What it does
- Serves boxed prediction images (e.g., *_pred.jpg) from outputs/<model>_<batch>/ as a 2-column grid
- Auto-detects available (model, batch) from ENF files in <enf-root> (e.g., yolov9t.enf, yolov9t_4b.enf)
- Picks outputs directory outputs/<model>_<batch> if it exists
- Parses latest logs/performance_result_*.log and shows a horizontal summary table under the model_batch selector
  - Converts latencies from ms -> s/img for clarity

Usage examples
  # from warboy-vision-models/warboy-vision-models:
  python viewer.py --host 0.0.0.0 --port 9999 \
    --enf-root ../models/enf/object_detection \
    --outputs-root outputs \
    --logs-root logs

  # force a specific model/batch:
  python viewer.py --model yolov9t --batch 1 --host 0.0.0.0 --port 9999 \
    --enf-root ../models/enf/object_detection --outputs-root outputs --logs-root logs
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, quote, unquote


# ENF naming: foo.enf (batch=1), foo_4b.enf (batch=4)
ENF_RE = re.compile(r"^(?P<model>.+?)(?:_(?P<batch>\d+)b)?\.enf$")

# performance_result_YYYYMMDD_HHMMSS.log
RESULT_LOG_RE = re.compile(r"^performance_result_(\d{8}_\d{6})\.log$")


def parse_enf_name(enf_path: Path) -> tuple[str, int] | None:
    m = ENF_RE.match(enf_path.name)
    if not m:
        return None
    model = m.group("model")
    batch_s = m.group("batch")
    batch = int(batch_s) if batch_s else 1
    return model, batch


def list_variants(enf_root: Path, outputs_root: Path) -> list[dict]:
    """
    Returns list of variants that exist in outputs:
      [{model, batch, enf, outdir, mtime}, ...]
    Sorted by outdir mtime desc (latest first).
    """
    variants: list[dict] = []
    if not enf_root.exists():
        return variants

    for enf in sorted(enf_root.glob("*.enf")):
        parsed = parse_enf_name(enf)
        if not parsed:
            continue
        model, batch = parsed
        outdir = outputs_root / f"{model}_{batch}"
        if not outdir.exists() or not outdir.is_dir():
            continue
        mtime = outdir.stat().st_mtime
        variants.append(
            {
                "model": model,
                "batch": batch,
                "enf": enf,
                "outdir": outdir,
                "mtime": mtime,
            }
        )

    variants.sort(key=lambda x: x["mtime"], reverse=True)
    return variants


def pick_variant(variants: list[dict], model: str | None, batch: int | None) -> dict | None:
    if not variants:
        return None
    if model is None and batch is None:
        return variants[0]
    for v in variants:
        if model is not None and v["model"] != model:
            continue
        if batch is not None and v["batch"] != batch:
            continue
        return v
    return None


def list_images(outdir: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    imgs = [p for p in outdir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # prefer *_pred.jpg first, then by name
    imgs.sort(key=lambda p: (0 if p.name.endswith("_pred.jpg") else 1, p.name))
    return imgs


def newest_result_log(logs_root: Path) -> Path | None:
    if not logs_root.exists():
        return None
    candidates = []
    for p in logs_root.iterdir():
        if not p.is_file():
            continue
        if RESULT_LOG_RE.match(p.name):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_result_log_for(model: str, batch: int, log_path: Path) -> dict | None:
    """
    Parse the summary row for given model/batch from performance_result_*.log.

    Expected row format:
    | 1 | 55.296 | 88.930 | 1.793 | 11.245 | 5.047 | 0.346 | 0.345 | success | 83.300 |
    """
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|"
        r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
        r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([A-Za-z_]+)\s*\|"
        r"(?:\s*([0-9.]+)\s*\|)?$"
    )


    model_hdr = f"[model : {model}]"
    in_model_block = False

    for line in lines:
        s = line.strip()
        if s.startswith("[model :"):
            in_model_block = s.startswith(model_hdr)
            continue
        if not in_model_block:
            continue

        m = row_re.match(s)
        if not m:
            continue

        b = int(m.group(1))
        if b != batch:
            continue

        e2e_active = float(m.group(2))
        infer_only = float(m.group(3))
        lat_pre_ms = float(m.group(4))
        lat_infer_ms = float(m.group(5))
        lat_post_ms = float(m.group(6))
        map_ = float(m.group(7))
        target = float(m.group(8))
        status = m.group(9)
        #sec = float(m.group(10))

        return {
            "batch_size": b,
            "e2e_active_img_s": e2e_active,
            "infer_only_img_s": infer_only,
            "lat_pre_ms": lat_pre_ms,
            "lat_infer_ms": lat_infer_ms,
            "lat_post_ms": lat_post_ms,
            "lat_pre_s_img": lat_pre_ms / 1000.0,
            "lat_infer_s_img": lat_infer_ms / 1000.0,
            "lat_post_s_img": lat_post_ms / 1000.0,
            "mAP": map_,
            "Target": target,
            "Status": status,
            #"sec_s": sec,
            "log_path": str(log_path),
        }

    return None


def html_page(title: str, variants: list[dict], current: dict | None, images: list[Path], summary: dict | None) -> str:
    # dropdown options
    options = []
    for v in variants:
        label = f"{v['model']}_{v['batch']}"
        selected = ""
        if current and v["model"] == current["model"] and v["batch"] == current["batch"]:
            selected = " selected"
        options.append(f'<option value="{html.escape(label)}"{selected}>{html.escape(label)}</option>')

    # summary table
    summary_html = ""
    if current:
        if summary:
            summary_html = f"""
            <div class="metrics-wrap">
              <div class="meta" style="margin: 0 0 8px 0;">
                metrics from: {html.escape(summary["log_path"])}
              </div>
              <div style="overflow-x:auto;">
                <table class="metrics">
                  <thead>
                    <tr>
                      <th>model_batch</th>
                      <th>e2e_active (img/s)</th>
                      <th>infer_only (img/s)</th>
                      <th>pre (s/img)</th>
                      <th>infer (s/img)</th>
                      <th>post (s/img)</th>
                      <th>mAP</th>
                      <th>Target</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{html.escape(current["model"])}_{current["batch"]}</td>
                      <td>{summary["e2e_active_img_s"]:.3f}</td>
                      <td>{summary["infer_only_img_s"]:.3f}</td>
                      <td>{summary["lat_pre_s_img"]:.6f}</td>
                      <td>{summary["lat_infer_s_img"]:.6f}</td>
                      <td>{summary["lat_post_s_img"]:.6f}</td>
                      <td>{summary["mAP"]:.3f}</td>
                      <td>{summary["Target"]:.3f}</td>
                      <td>{html.escape(summary["Status"])}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            """
        else:
            summary_html = """
            <div class="metrics-wrap">
              <div class="empty">
                성능 로그(performance_result_*.log)에서 해당 model/batch 행을 찾지 못했습니다.
                (logs-root 경로, model명, batch, 로그 형식 확인)
              </div>
            </div>
            """

    # image cards
    cards = []
    for img in images:
        url = "/img/" + quote(img.name)
        cards.append(
            f"""
            <div class="card">
              <a href="{url}" target="_blank" rel="noopener">
                <img src="{url}" loading="lazy" />
              </a>
              <div class="cap">{html.escape(img.name)}</div>
            </div>
            """
        )

    current_label = f"{current['model']}_{current['batch']}" if current else "(none)"
    outdir_txt = str(current["outdir"]) if current else "(none)"

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0; padding: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      background: #0b0f17; color: #e8eefc;
    }}
    header {{
      position: sticky; top: 0;
      background: rgba(11,15,23,0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid rgba(232,238,252,0.12);
      padding: 12px 14px;
      z-index: 10;
    }}
    .row {{
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    }}
    .title {{
      font-weight: 700; font-size: 16px;
    }}
    .meta {{
      opacity: 0.85; font-size: 12px;
    }}
    select, button {{
      background: #121a2a; color: #e8eefc;
      border: 1px solid rgba(232,238,252,0.18);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
    }}
    button:hover {{
      cursor: pointer;
      border-color: rgba(232,238,252,0.35);
    }}
    main {{
      padding: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr)); /* 2 columns */
      gap: 12px;
    }}
    .card {{
      background: #0f1627;
      border: 1px solid rgba(232,238,252,0.12);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    .card img {{
      width: 100%;
      height: auto;
      display: block;
      background: #0b0f17;
    }}
    .cap {{
      padding: 8px 10px;
      font-size: 12px;
      opacity: 0.9;
      word-break: break-all;
      border-top: 1px solid rgba(232,238,252,0.08);
    }}
    .empty {{
      padding: 16px;
      border: 1px dashed rgba(232,238,252,0.25);
      border-radius: 14px;
      opacity: 0.9;
      background: rgba(15,22,39,0.6);
    }}

    .metrics-wrap {{
      margin-top: 12px;
    }}
    table.metrics {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      background: #0f1627;
      border: 1px solid rgba(232,238,252,0.12);
      border-radius: 14px;
      overflow: hidden;
      font-size: 13px;
    }}
    table.metrics th {{
      text-align: left;
      font-size: 12px;
      opacity: 0.95;
      padding: 10px;
      border-bottom: 1px solid rgba(232,238,252,0.08);
      white-space: nowrap;
    }}
    table.metrics td {{
      padding: 10px;
      border-bottom: 1px solid rgba(232,238,252,0.06);
      white-space: nowrap;
    }}
    table.metrics tbody tr:last-child td {{
      border-bottom: none;
    }}

    @media (max-width: 720px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="row">
      <div class="title">Warboy Outputs Viewer</div>
      <div class="meta">Current: <b>{html.escape(current_label)}</b> · Dir: {html.escape(outdir_txt)}</div>
    </div>
    <div class="row" style="margin-top:10px;">
      <label class="meta">model_batch:</label>
      <select id="variant">
        {''.join(options)}
      </select>
      <button onclick="applyVariant()">Apply</button>
      <button onclick="location.reload()">Refresh</button>
      <div class="meta">Images: {len(images)}</div>
    </div>

    {summary_html}
  </header>

  <main>
    {"<div class='empty'>표시할 이미지가 없습니다. Warboy 실행 후 outputs/&lt;model&gt;_&lt;batch&gt;/에 *_pred.jpg가 생성됐는지 확인하세요.</div>" if len(images)==0 else f"<div class='grid'>{''.join(cards)}</div>"}
  </main>

  <script>
    function applyVariant() {{
      const v = document.getElementById('variant').value;
      // v format: model_batch (model may contain underscores)
      const parts = v.split('_');
      const batch = parts.pop();
      const model = parts.join('_');
      const url = new URL(window.location.href);
      url.searchParams.set('model', model);
      url.searchParams.set('batch', batch);
      window.location.href = url.toString();
    }}
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "WarboyViewer/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        app = self.server.app  # type: ignore

        if path == "/" or path == "/index.html":
            model = qs.get("model", [None])[0]
            batch_s = qs.get("batch", [None])[0]
            batch = int(batch_s) if batch_s and batch_s.isdigit() else None

            variants = app["variants_fn"]()
            current = pick_variant(variants, model, batch)
            images = list_images(current["outdir"]) if current else []

            summary = None
            if current:
                log_path = newest_result_log(app["logs_root"])
                if log_path:
                    summary = parse_result_log_for(current["model"], current["batch"], log_path)

            body = html_page("Warboy Outputs Viewer", variants, current, images, summary).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # serve image file: /img/<filename>
        if path.startswith("/img/"):
            filename = unquote(path[len("/img/"):])

            variants = app["variants_fn"]()
            model = qs.get("model", [None])[0]
            batch_s = qs.get("batch", [None])[0]
            batch = int(batch_s) if batch_s and batch_s.isdigit() else None
            current = pick_variant(variants, model, batch)
            if not current:
                self.send_error(404, "No outputs directory found")
                return

            file_path = (current["outdir"] / filename).resolve()
            # prevent path traversal
            if current["outdir"].resolve() not in file_path.parents:
                self.send_error(403, "Forbidden")
                return
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "File not found")
                return

            ctype, _ = mimetypes.guess_type(str(file_path))
            ctype = ctype or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=9999, help="Port (default: 9999)")
    ap.add_argument("--enf-root", default="../models/enf/object_detection", help="ENF directory")
    ap.add_argument("--outputs-root", default="outputs", help="Outputs root directory")
    ap.add_argument("--model", default=None, help="Force model name (optional)")
    ap.add_argument("--batch", type=int, default=None, help="Force batch size (optional)")
    ap.add_argument("--logs-root", default="logs", help="Logs directory (performance_result_*.log)")
    args = ap.parse_args()
    
    enf_root = Path(args.enf_root)
    outputs_root = Path(args.outputs_root)
    logs_root = Path(args.logs_root)

    def variants_fn():
        return list_variants(enf_root, outputs_root)

    variants = variants_fn()
    cur = pick_variant(variants, args.model, args.batch)
    if not cur:
        print(f"[WARN] No outputs found.")
        print(f"       ENF root: {enf_root}")
        print(f"       Outputs root: {outputs_root}")
        print("       Run Warboy first so outputs/<model>_<batch>/ has images, then refresh.")
    else:
        print(f"[INFO] Selected: {cur['model']}_{cur['batch']}  dir={cur['outdir']}")

    if newest_result_log(logs_root):
        print(f"[INFO] Logs root OK: {logs_root}")
    else:
        print(f"[WARN] No performance_result_*.log found under logs root: {logs_root}")

    httpd = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    httpd.app = {  # type: ignore
        "variants_fn": variants_fn,
        "logs_root": logs_root,
    }
    print(f"[INFO] Serving on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
