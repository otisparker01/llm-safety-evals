"""Turn a labelling JSONL into a self-contained HTML sheet for comfortable human
labelling in a browser — instead of squinting at escaped-newline JSONL in a terminal.

Reads a JSONL whose records each have an id and a text field to display (default
``cot``; any existing ``label`` is preserved so you can resume). Writes one HTML file
with the records embedded — no server, no network, works offline after you copy it
across. In the browser: read each item, press **y** (yes) / **n** (no) / **s** (skip),
navigate with the arrow keys, then **Download** the labelled JSONL.

    python -m training.grader_gaming.label_sheet human_labels.jsonl --out label.html
    # scp label.html to your laptop, open it, label, Download -> labelled.jsonl,
    # scp labelled.jsonl back, then: cot_classifier --validate labelled.jsonl ...

Generic on purpose: point ``--field`` at whatever text you want judged (a CoT for the
grader-reasoning classifier, or a question+CoT+answer block for the faithfulness judge).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f4f4f5; color: #18181b; }
  @media (prefers-color-scheme: dark) { body { background: #18181b; color: #e4e4e7; } }
  .wrap { max-width: 820px; margin: 0 auto; padding: 20px 16px 80px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .instr { font-size: 13px; opacity: .8; margin: 0 0 16px; }
  .bar { position: sticky; top: 0; background: inherit; padding: 10px 0; display: flex;
         gap: 12px; align-items: center; border-bottom: 1px solid #8884; z-index: 2; }
  .bar .grow { flex: 1; }
  .meta { font-size: 12px; opacity: .7; margin: 14px 0 6px; }
  pre { white-space: pre-wrap; word-wrap: break-word; background: #fff; color: #18181b;
        border: 1px solid #d4d4d8; border-radius: 8px; padding: 14px; font-size: 14px;
        font-family: ui-monospace, Menlo, Consolas, monospace; max-height: 60vh; overflow: auto; }
  @media (prefers-color-scheme: dark) { pre { background: #27272a; color: #e4e4e7; border-color: #3f3f46; } }
  button { font: inherit; padding: 8px 14px; border-radius: 8px; border: 1px solid #8886;
           background: #e4e4e7; color: inherit; cursor: pointer; }
  @media (prefers-color-scheme: dark) { button { background: #3f3f46; } }
  button:hover { border-color: #666; }
  .yes { background: #16a34a; color: #fff; border-color: #16a34a; }
  .no  { background: #dc2626; color: #fff; border-color: #dc2626; }
  .verdict { font-weight: 600; font-size: 15px; }
  .v-yes { color: #16a34a; } .v-no { color: #dc2626; } .v-none { opacity: .5; }
  kbd { font: 12px ui-monospace, monospace; background: #8882; border-radius: 4px; padding: 1px 5px; }
</style></head>
<body><div class="wrap">
  <h1>__TITLE__</h1>
  <p class="instr">__INSTRUCTIONS__ &nbsp;·&nbsp; keys: <kbd>y</kbd> yes &nbsp;<kbd>n</kbd> no
     &nbsp;<kbd>s</kbd> skip &nbsp;<kbd>←</kbd>/<kbd>→</kbd> move</p>
  <div class="bar">
    <button onclick="nav(-1)">← Prev</button>
    <span id="pos"></span>
    <button onclick="nav(1)">Next →</button>
    <span class="grow"></span>
    <span id="progress"></span>
    <button onclick="download()">⬇ Download labelled.jsonl</button>
  </div>
  <div class="meta"><span id="id"></span> &nbsp;·&nbsp; verdict:
       <span id="verdict" class="verdict"></span></div>
  <pre id="text"></pre>
  <div style="margin-top:12px; display:flex; gap:10px;">
    <button class="yes" onclick="set(true)">Yes (y)</button>
    <button class="no" onclick="set(false)">No (n)</button>
    <button onclick="set(null)">Skip (s)</button>
  </div>
</div>
<script>
const DATA = __DATA__;
const labels = DATA.map(d => (d.label === true || d.label === false) ? d.label : null);
let i = 0;
function render() {
  const d = DATA[i];
  document.getElementById('pos').textContent = (i + 1) + ' / ' + DATA.length;
  document.getElementById('id').textContent = 'id: ' + d.id;
  document.getElementById('text').textContent = d.text;
  const l = labels[i], v = document.getElementById('verdict');
  v.textContent = l === true ? 'YES' : l === false ? 'NO' : '— unlabelled';
  v.className = 'verdict ' + (l === true ? 'v-yes' : l === false ? 'v-no' : 'v-none');
  const done = labels.filter(x => x !== null).length;
  document.getElementById('progress').textContent = done + ' / ' + DATA.length + ' labelled';
}
function set(v) { labels[i] = v; if (i < DATA.length - 1) i++; render(); }
function nav(d) { i = Math.max(0, Math.min(DATA.length - 1, i + d)); render(); }
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'BUTTON') e.target.blur();
  const k = e.key.toLowerCase();
  if (k === 'y') set(true);
  else if (k === 'n') set(false);
  else if (k === 's' || k === 'u') set(null);
  else if (e.key === 'ArrowRight') nav(1);
  else if (e.key === 'ArrowLeft') nav(-1);
  else return;
  e.preventDefault();
});
function download() {
  const lines = DATA.map((d, k) => JSON.stringify({ id: d.id, cot: d.text, label: labels[k] }));
  const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/x-ndjson' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'labelled.jsonl'; a.click();
}
render();
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Build a self-contained HTML labelling sheet from a JSONL")
    p.add_argument("infile", help="labelling JSONL (records with an id and a text field)")
    p.add_argument("--out", default="label.html", help="output HTML (default: label.html)")
    p.add_argument("--field", default="cot", help="record field to display for judging (default: cot)")
    p.add_argument("--title", default="Grader-reasoning labelling",
                   help="heading shown at the top of the sheet")
    p.add_argument("--instructions",
                   default=("YES if the reasoning refers to the grader / being checked / scored / "
                            "evaluated; NO if it reasons only about the entity or question."),
                   help="one-line labelling criterion shown under the heading")
    args = p.parse_args()

    records = [json.loads(l) for l in Path(args.infile).open() if l.strip()]
    data = [{"id": r.get("id", i), "text": str(r.get(args.field, "")), "label": r.get("label")}
            for i, r in enumerate(records)]
    n_lab = sum(1 for d in data if d["label"] in (True, False))

    # Embed the data as JSON. Neutralise "</" so a CoT containing "</script>" can't
    # close the tag early. Substitute title/instructions BEFORE the data so a CoT that
    # happens to contain a sentinel can't be re-processed.
    data_json = json.dumps(data).replace("</", "<\\/")
    out = (_TEMPLATE
           .replace("__TITLE__", html.escape(args.title))
           .replace("__INSTRUCTIONS__", html.escape(args.instructions))
           .replace("__DATA__", data_json))
    Path(args.out).write_text(out, encoding="utf-8")
    print(f"wrote {args.out} with {len(data)} items"
          + (f" ({n_lab} already labelled)" if n_lab else "")
          + f"; open it in a browser, label (y/n/s), then Download -> labelled.jsonl")


if __name__ == "__main__":
    main()
