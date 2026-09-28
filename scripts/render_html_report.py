"""Build the standalone HTML report using public evidence only."""
from pathlib import Path
import html
import json
import re
from evaluate_product_reviews import NAMES, LABELS, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/benchmarks/product-review-sentiment/data"


def esc(value):
    return html.escape(str(value), quote=True)


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def main():
    cases = load("dataset.json")["cases"]
    records = [json.loads(line) for line in (DATA / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = load("summary.json")
    assert summarize(cases, records) == summary
    by_id = {r["id"]: r for r in records}
    assert len(cases) == len(records) == 100
    rows = []
    for c in cases:
        r = by_id[c["id"]]
        assert r["correct"] == (r["predicted"] == c["expected"])
        verdict = "一致" if r["correct"] else "不一致"
        rows.append(f'''<tr data-id="{esc(c['id'])}" data-expected="{esc(c['expected'])}" data-verdict="{'match' if r['correct'] else 'miss'}" data-latency="{r['client_ms']}">
<th scope="row">{esc(c['id'])}</th><td class="review"><span class="product">{esc(c['product'])}</span><p>{esc(c['text'])}</p><details><summary>期待値の根拠</summary><p>{esc(c['rationale'])}</p></details></td><td>{NAMES[c['expected']]}</td><td>{NAMES[r['predicted']]}</td><td class="{'match' if r['correct'] else 'miss'}">{verdict}</td><td class="number">{r['client_ms']:.2f}</td><td class="number">{r['prediction_ms']:.2f}</td></tr>''')
    confusion = []
    for gold in LABELS:
        confusion.append('<tr><th scope="row">'+NAMES[gold]+'</th>'+''.join(
            f'<td class="{"diagonal" if gold == pred else "off-diagonal"}" style="--strength:{summary["confusion"][gold][pred]/34:.3f}"><strong>{summary["confusion"][gold][pred]}</strong></td>' for pred in LABELS)+'</tr>')
    per_class = []
    for label in LABELS:
        m = summary["per_class"][label]
        per_class.append(f'<tr><th scope="row">{NAMES[label]}</th><td>{m["correct"]}/{m["support"]}</td><td>{m["recall"]*100:.1f}%</td><td>{m["precision"]*100:.1f}%</td><td>{m["f1"]:.3f}</td></tr>')
    examples = []
    for case_id in ("R040", "R038", "R100"):
        c = next(c for c in cases if c["id"] == case_id)
        r = by_id[case_id]
        examples.append(f'<article class="example"><div class="example-label">{case_id} / {esc(c["product"])}</div><blockquote>{esc(c["text"])}</blockquote><p><span>期待：{NAMES[c["expected"]]}</span> <span class="miss">判定：{NAMES[r["predicted"]]}</span></p><p class="small">{esc(c["rationale"])}</p></article>')
    speed = []
    text = (ROOT / "SPEED_VALIDATION.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if re.match(r"\| (短文|中文|長文) \|", line):
            v = [part.strip() for part in line.strip('|').split('|')]
            speed.append(f'<tr><th scope="row">{v[0]}・{v[1]}字</th><td>{v[3]}</td><td>{v[4]}</td><td class="bar-cell"><div class="bar" style="width:{float(v[5])/4500*100:.2f}%"></div><span>{float(v[5]):,.1f}</span></td><td class="number">{float(v[6]):,.1f}</td><td class="number">{float(v[7]):,.1f}</td></tr>')
    assert len(speed) == 9
    inquiry = []
    for line in (ROOT / "ACCURACY_VALIDATION.md").read_text(encoding="utf-8").splitlines():
        if re.match(r"\| [^|]+ \| \d+/60 \|", line):
            v = [part.strip() for part in line.strip('|').split('|')]
            inquiry.append(f'<tr><th scope="row">{esc(v[0])}</th><td>{esc(v[1])}</td><td>{esc(v[2])}</td></tr>')
    assert len(inquiry) == 8
    fixture = json.loads((ROOT / "examples/benchmark-ja.json").read_text(encoding="utf-8"))
    states = ''.join(f'<article><h4>{esc(s["label"])}・{len(s["text"])}文字</h4><p class="preserve">{esc(s["text"])}</p></article>' for s in fixture['states'])
    replacements = {
        'ROWS': '\n'.join(rows), 'CONFUSION': '\n'.join(confusion),
        'PER_CLASS': '\n'.join(per_class), 'EXAMPLES': '\n'.join(examples),
        'SPEED_ROWS': '\n'.join(speed), 'INQUIRY_ROWS': '\n'.join(inquiry), 'SPEED_TEXTS': states,
        'QUESTION': esc(json.dumps(load('question.json'), ensure_ascii=False, indent=2)),
        'ACCURACY': f'{summary["accuracy"]*100:.1f}', 'CORRECT': str(summary['correct']),
        'MACRO_F1': f'{summary["macro_f1"]:.3f}',
        'MEDIAN': f'{summary["client_latency_ms"]["p50"]:.2f}',
        'P95': f'{summary["client_latency_ms"]["p95"]:.2f}',
        'SERVER_MEDIAN': f'{summary["prediction_latency_ms"]["p50"]:.2f}',
        'MIN': f'{summary["client_latency_ms"]["min"]:.2f}',
        'MAX': f'{summary["client_latency_ms"]["max"]:.2f}',
    }
    output = (ROOT / 'scripts/report_template.html').read_text(encoding='utf-8')
    for key, value in replacements.items():
        output = output.replace('@@'+key+'@@', value)
    assert not re.search(r'@@[A-Z_]+@@', output)
    (ROOT / 'docs/report.html').write_text(output, encoding='utf-8')
    print('HTML generated from public evidence: 100 review rows, 9 speed conditions, 8 inquiry metrics.')


if __name__ == '__main__':
    main()
