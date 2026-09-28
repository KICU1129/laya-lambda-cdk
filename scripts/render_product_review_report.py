"""Render a public Markdown report from fixed cases and measured API responses."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from evaluate_product_reviews import LABELS, NAMES, summarize


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(n):
    return f"{n * 100:.1f}%"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def ms(value):
    return f"{value:.2f}" if value is not None else "—"


def render(root):
    run = root / "data"
    dataset, manifest, summary = (read(run / name) for name in ("dataset.json", "manifest.json", "summary.json"))
    cases = dataset["cases"]
    records = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    # This publication template requires a completed run. Partial/failing runs
    # still have summary.json, where unattempted cases remain in the denominator.
    if len(records) != len(cases) or not any(r["success"] for r in records):
        raise ValueError("Markdown requires 100 attempted cases and at least one valid response; inspect summary.json for partial/failed runs")
    expected_by_id = {c["id"]: c["expected"] for c in cases}
    for r in records:
        correct = r["success"] and r["predicted"] == expected_by_id[r["id"]]
        if r["expected"] != expected_by_id[r["id"]] or r["correct"] != correct:
            raise ValueError("Saved expected/correct fields disagree with frozen dataset and prediction")
    if summarize(cases, records) != summary:
        raise ValueError("Saved metrics do not match records")
    for filename, key in (("dataset.json", "dataset_sha256"), ("question.json", "question_sha256")):
        if hashlib.sha256((run / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("Frozen fixture or question was modified")
    by_id = {r["id"]: r for r in records}
    latency = summary["client_latency_ms"]
    model_latency = summary["prediction_latency_ms"]
    model = next(r for r in records if r["success"])
    warmup = read(run / "warmup.json")
    errors = [c for c in cases if not by_id.get(c["id"], {}).get("correct", False)]
    errors.sort(key=lambda c: -max(by_id[c["id"]].get("probabilities", {}).values(), default=0))
    confusions = sorted(((summary["confusion"][g][p], g, p) for g in LABELS for p in LABELS if g != p), reverse=True)
    lines = ["# 日本語の商品レビュー100件で検証する Laya の感情分類", "",
             f"**正答率は{pct(summary['accuracy'])}（{summary['correct']}/100件）、Macro-F1は{summary['macro_f1']:.3f}でした。初期化後のAPI応答時間は中央値{latency['p50']:.2f}ms、p95は{latency['p95']:.2f}msでした。**", "",
             "架空の商品レビュー100件を「肯定・中立・否定」に分類する、日本語の小規模な検証です。期待値は推論前に固定しました。以下の値はこの問題セットに対する成績であり、実際の商品レビュー全体の精度を推定するものではありません。", "",
             "## 1. 結論と主要結果", "",
             "| 指標 | 実測値 | 読み方 |", "|---|---:|---|",
             f"| 正答率 | {summary['correct']}/100（{pct(summary['accuracy'])}） | 期待する感情とモデルの選択が一致した割合。失敗・未取得も分母に含む |",
             f"| Macro-F1 | {summary['macro_f1']:.3f} | 3クラスのF1を同じ重みで平均。最大1 |",
             f"| API応答成功 | {summary['successful']}/100 | HTTP 200かつ有効な3分類の回答 |",
             f"| 多数派固定の参考値 | {pct(summary['majority_baseline'])} | 全件を最多クラスの「肯定」とした場合 |",
             f"| API応答時間・中央値 / p95 | {latency['p50']:.2f} / {latency['p95']:.2f} ms | クライアントから観測した、通信を含む時間 |",
             f"| API応答時間・平均 / 最小 / 最大 | {latency['mean']:.2f} / {latency['min']:.2f} / {latency['max']:.2f} ms | 成功した{latency['n']}件 |",
             f"| サーバー推論処理時間・中央値 / p95 | {model_latency['p50']:.2f} / {model_latency['p95']:.2f} ms | APIが返した prediction_ms |", ""]
    if confusions[0][0]:
        count, gold, predicted = confusions[0]
        lines += [f"最も多い取り違えは、期待値「{NAMES[gold]}」を「{NAMES[predicted]}」とする{count}件でした。以下のクラス別成績と実例を併せて評価してください。", ""]
    negative = summary["per_class"]["negative"]
    if negative["predicted"] > negative["support"]:
        lines += [f"**評価：今回の問題セットでは否定に寄る傾向がありました。** モデルは{negative['predicted']}/100件を否定とし、そのうち{negative['predicted'] - negative['correct']}件は期待値が肯定または中立でした。肯定の再現率は{pct(summary['per_class']['positive']['recall'])}、中立は{pct(summary['per_class']['neutral']['recall'])}です。否定の再現率だけでは3分類全体の性能を評価できません。", ""]
    lines += ["**この応答時間にはモデル初期化を含みません。** Provisioned Concurrencyを1にして準備完了を待ち、別の短文で1回ウォームアップした後に測定しました。初回アクセスの速度や同時アクセスへの耐性は、この100件から判断できません。", "",
              "## 2. どの感情を判定できたか", "", "| 期待する感情 | 件数 | 正解数 | 再現率 | 適合率 | F1 | モデルが選んだ件数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        c = summary["per_class"][label]
        lines += [f"| {NAMES[label]} | {c['support']} | {c['correct']} | {pct(c['recall'])} | {pct(c['precision'])} | {c['f1']:.3f} | {c['predicted']} |"]
    lines += ["", "再現率は、その感情が期待される問題のうち正解した割合です。適合率は、モデルがその感情を選んだ問題のうち正しかった割合です。選択が0件の場合、集計上の適合率・F1は0とします。", "",
              "### 混同行列", "", "行が期待値、列がモデルの判定です。対角線が正解です。", "",
              "| 期待値 ＼ 判定 | 肯定 | 中立 | 否定 | 失敗・未取得 |", "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        lines += [f"| {NAMES[label]} | " + " | ".join(str(summary["confusion"][label][p]) for p in (*LABELS, "error")) + " |"]
    lines += ["", "## 3. 誤判定の具体例", "",
              "誤判定のうち、モデルが選んだ答えの確率が高い順に最大5件を掲載します。この確率は実際の正答率を保証する値ではありません。全問題は[100件の結果表](RESULTS.md)に掲載しています。", "",
              "| ID | レビュー全文 | 期待値 | 実際の判定 | 選択した答えの確率 | 期待値の根拠 |", "|---|---|---|---|---:|---|"]
    for c in errors[:5]:
        r = by_id[c["id"]]
        lines += [f"| {c['id']} | {cell(c['text'])} | {NAMES[c['expected']]} | {NAMES.get(r['predicted'], '失敗')} | {pct(max(r['probabilities'].values(), default=0))} | {cell(c['rationale'])} |"]
    if not errors:
        lines += ["| — | 今回の問題セットでは誤判定なし | — | — | — | — |"]
    lines += ["", "## 4. 評価方法", "", "### 問題と期待値", "",
              "- 実在の投稿を転載せず、この検証のためにAIで作成した日本語の架空レビュー100件を使用しました。商品名も一般的な種類名です。",
              "- 肯定34件、中立33件、否定33件。肯定・否定の表現だけでなく、事実記述、長所と短所の混在、否定表現などを含めました。",
              "- 作成担当とは別のAIによるラベル確認を推論前に実施し、期待値・根拠・問題文を固定しました。人による独立アノテーションは行っていません。",
              "- 全体の評価が肯定優勢なら肯定、否定優勢なら否定、評価を含まない事実記述または賛否同程度なら中立としました。個別の期待値と理由はデータセットに収録しています。",
              "- モデルに渡した入力はレビュー本文と下記の共通質問だけです。期待値、根拠、ID、商品カテゴリは渡していません。", "",
              "### モデルに送った質問", "", "```json", json.dumps(read(run / "question.json"), ensure_ascii=False, indent=2), "```", "",
              "### 実行・採点・時間計測", "",
              "1. データセットと質問のSHA-256を記録して固定しました。",
              "2. Provisioned Concurrencyを1に設定し、READYを待ちました。別の短文で1回ウォームアップし、その応答は本集計から除外しました。",
              "3. seed=20260928で順序を並べ替え、100件を直列に各1回送信しました。各リクエスト後に0.6秒待ち、自動再試行は行っていません。",
              "4. APIのchoiceを採用し、返された確率の最大候補と整合することを検査しました。正解ラベルの変更、閾値調整、質問のチューニングは行っていません。",
              "5. API応答時間はHTTP送信開始直前から応答本文の受信完了までです。TLS接続など通信時間を含み、署名作成・JSON解析・待機時間を除きます。urllibで毎回接続し、接続再利用の最適化はしていません。",
              "6. サーバー推論処理時間はmeta.prediction_msです。ハンドラーの計測範囲であるトークン予算検査とモデル推論を含み、モデルロードとネットワーク時間は含みません。",
              "7. p95は昇順データの位置(n−1)×0.95で線形補間しました。精度は全100件、レイテンシーの主集計は有効な回答が得られた件が対象です。", "",
              "### 検証条件", "", "| 項目 | 内容 |", "|---|---|",
              "| 経路 | ローカルのPythonクライアント → API Gateway HTTP API → Lambda → CPU推論 |",
              f"| モデル | {model['model']} |", f"| モデルrevision | `{model['model_revision']}` |",
              f"| Laya | {model['laya_version']} |",
              f"| ウォームアップ応答時間 | {warmup['client_ms']:.2f}ms（100件に含めない） |",
              f"| レビュー文字数 | {min(len(c['text']) for c in cases)}〜{max(len(c['text']) for c in cases)}文字、中央値{statistics.median(len(c['text']) for c in cases):.0f}文字 |", "",
              "実行環境の詳細は非公開です。保存した結果から集計を再計算できますが、同じレイテンシーの再現を目的とした資料ではありません。", "",
              "## 5. この検証で分かる範囲", "",
              "今回の100件・日本語の質問文・選択肢・固定モデル・実行環境での分類成績と、初期化後の応答時間を確認できます。以前の問い合わせ8項目の評価とは問題も出力形式も異なるため、その正答率と直接比較して改善・悪化とは判断しません。", "",
              "合成レビューは実データの無作為標本ではなく、クラス比率もほぼ均等に設計しています。人間のラベル一致率は未測定です。100件各1回のため、同一入力での再現性、時間帯による変動、長文、皮肉全般、他モデルとの差、並列負荷は評価していません。実運用での精度を示す信頼区間は付けていません。", "",
              "文章は短く整っており、中立には仕様の説明や中間評価を明示した例が多く含まれます。同じ商品の肯定・否定の組も多いため、100件が100種類の独立した話題を意味するわけではありません。誤字・絵文字・方言・配送と商品への感情の混在も十分に網羅していません。利点と難点が均衡する例を中立に含めるのは今回の分類規約です。", "",
              "## 6. データと再実行", "",
              "| ファイル | 内容 |", "|---|---|",
              "| [RESULTS.md](RESULTS.md) | 全100問の全文、期待値、判定、一致／不一致、各レイテンシー |",
              "| [data/dataset.json](data/dataset.json) | 事前に固定した問題・期待値・理由 |",
              "| [data/question.json](data/question.json) | 実際に使用した共通質問 |",
              "| [data/results.jsonl](data/results.jsonl) | 実行順の判定・確率・時間。HTTP応答から判定と計測値を抜粋 |",
              "| [data/summary.json](data/summary.json) | 正答率、クラス別指標、混同行列、レイテンシー集計 |",
              "| [data/manifest.json](data/manifest.json) | 測定手順、ハッシュ、実行順 |",
              "| [data/warmup.json](data/warmup.json) | 集計から除外したウォームアップ |",
              "| [data/label-review.json](data/label-review.json) | 推論前に別のAIが全100件を確認した記録 |",
              "| [scripts/](scripts/) | API評価とMarkdown生成のコード |", "",
              "### 保存データだけでMarkdownを再生成", "", "```bash", "python scripts/render_product_review_report.py --root .", "```", "",
              "### 自分のAWS環境で再評価", "",
              "このLambda実装をデプロイし、手元だけにCDK出力を用意してください。新しい結果ディレクトリを指定します。実行中はPC=1を作成し、finallyで削除・確認するためAWS利用料が発生します。既存のPC設定がある場合は上書きせず停止します。", "", "```bash",
              "python -m pip install -r requirements.txt",
              "python scripts/evaluate_product_reviews.py --dataset data/dataset.json --outputs PRIVATE_OUTPUTS_JSON --out-dir new-run --profile YOUR_PROFILE",
              "```", "",
              "計測結果はnew-runに保存されます。途中停止した場合は、対象aliasのProvisioned Concurrencyが解除されているか確認してください。", ""]
    lines += ["Markdownの生成は100件の送信が完了し、有効な応答が1件以上ある実行が対象です。途中停止や全件失敗の場合はsummary.jsonを確認してください。途中停止時も未取得分を含む全100件を正答率の分母に残します。", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    details = ["# 日本語商品レビュー100件：全問題と判定結果", "", "[結論・集計・評価方法に戻る](README.md)", "",
               "各レビューに共通して「商品に対する全体的な感情」を肯定・中立・否定から選ばせました。期待値はAPI実行前に固定したラベルです。「一致」が正解、「不一致」が誤りです。全文を省略せず、問題ID順で掲載しています。実行順はmanifest.jsonに記録しています。", "",
               "API時間はクライアントで観測した通信込みの時間、推論時間はサーバー側のprediction_msです。どちらもミリ秒（ms）。初期化・ウォームアップ・リクエスト間の待機は含みません。", "",
               "| ID | 商品／問題文（全文） | 期待する結果 | 実際の結果 | 期待値と一致するか | API時間 ms | 推論時間 ms |",
               "|---|---|---|---|---|---:|---:|"]
    for case in cases:
        r = by_id[case["id"]]
        details += [f"| {case['id']} | **{cell(case['product'])}**<br>{cell(case['text'])} | {NAMES[case['expected']]} | {NAMES.get(r['predicted'], '失敗')} | {'一致' if r['correct'] else '不一致'} | {ms(r['client_ms'])} | {ms(r['prediction_ms'])} |"]
    details += ["", "## 期待する結果の根拠", "", "感情の正解には解釈が入るため、採点に使った理由も公開します。この理由はモデルには送信していません。", "",
                "| ID | 期待値 | 事前に付けた理由 |", "|---|---|---|"]
    for case in cases:
        details += [f"| {case['id']} | {NAMES[case['expected']]} | {cell(case['rationale'])} |"]
    (root / "RESULTS.md").write_text("\n".join(details) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    render(parser.parse_args().root)
