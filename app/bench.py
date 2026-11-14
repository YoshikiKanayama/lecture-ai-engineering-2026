"""第1回 宿題（発展）: 応答時間とトークン数を、モデル・質問ごとに計測して比べる。

使い方:
    uv run python -m app.bench                                   # 既定のモデル・質問で3回ずつ
    uv run python -m app.bench --models gemini-3.8-flash --runs 2
    uv run python -m app.bench --interval 0 --runs 20            # 間隔なしで連投し、429 を観察する

結果は Markdown の表で出力する（answer.md にそのまま貼れる）。
"""

import argparse
import statistics
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors

from app.config import get_api_key

DEFAULT_MODELS = ["gemini-3.5-flash-lite"]
DEFAULT_QUESTIONS = [
    "LoRAとは何ですか？",
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "1+1は？数字だけで答えて",
]
# gemini-3.8-flash の無料枠（毎分5リクエスト）を超えない間隔
DEFAULT_INTERVAL_SEC = 13.0


@dataclass
class Result:
    model: str
    question: str
    run: int
    status: str  # "ok" か HTTP ステータスコード
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    total_tokens: int | None = None


def measure(client, model: str, question: str, run: int) -> Result:
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        return Result(model, question, run, str(e.code), time.perf_counter() - start)
    seconds = time.perf_counter() - start

    usage = response.usage_metadata
    if usage is None:
        return Result(model, question, run, "ok", seconds)
    return Result(
        model,
        question,
        run,
        "ok",
        seconds,
        prompt_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        thoughts_tokens=usage.thoughts_token_count,
        total_tokens=usage.total_token_count,
    )


def _mean(values: list[float | int | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def _fmt(value: float | int | None, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summarize(results: list[Result]) -> list[dict]:
    """モデル×質問ごとに、成功した回の平均と、エラーの回数をまとめる。"""
    groups: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        groups.setdefault((r.model, r.question), []).append(r)

    rows = []
    for (model, question), rs in groups.items():
        ok = [r for r in rs if r.status == "ok"]
        errors_ = [r.status for r in rs if r.status != "ok"]
        mean_sec = _mean([r.seconds for r in ok])
        mean_out = _mean([r.output_tokens for r in ok])
        rows.append(
            {
                "model": model,
                "question": question,
                "ok": len(ok),
                "errors": errors_,
                "mean_sec": mean_sec,
                "mean_prompt": _mean([r.prompt_tokens for r in ok]),
                "mean_output": mean_out,
                "mean_thoughts": _mean([r.thoughts_tokens for r in ok]),
                "mean_total": _mean([r.total_tokens for r in ok]),
                # 思考トークンは含めない（usage の出力トークンだけを生成速度の目安にする）
                "output_per_sec": mean_out / mean_sec if mean_out and mean_sec else None,
            }
        )
    return rows


def results_table(results: list[Result]) -> str:
    lines = [
        "| モデル | 質問 | 回 | 結果 | 秒 | 入力 | 出力 | 思考 | 合計 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.model} | {r.question} | {r.run} | {r.status} | {r.seconds:.2f} "
            f"| {_fmt(r.prompt_tokens)} | {_fmt(r.output_tokens)} "
            f"| {_fmt(r.thoughts_tokens)} | {_fmt(r.total_tokens)} |"
        )
    return "\n".join(lines)


def summary_table(rows: list[dict]) -> str:
    lines = [
        "| モデル | 質問 | 成功 | エラー | 平均秒 | 平均入力 | 平均出力 | 平均思考 | 平均合計 | 出力/秒 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in rows:
        lines.append(
            f"| {s['model']} | {s['question']} | {s['ok']} | {', '.join(s['errors']) or '-'} "
            f"| {_fmt(s['mean_sec'], 2)} | {_fmt(s['mean_prompt'])} | {_fmt(s['mean_output'])} "
            f"| {_fmt(s['mean_thoughts'])} | {_fmt(s['mean_total'])} | {_fmt(s['output_per_sec'], 1)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC)
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    results = []
    total = len(args.models) * len(args.questions) * args.runs
    for model in args.models:
        for question in args.questions:
            for run in range(1, args.runs + 1):
                if results:
                    time.sleep(args.interval)
                r = measure(client, model, question, run)
                results.append(r)
                print(f"[{len(results)}/{total}] {model} 回{run} {r.status} {r.seconds:.2f}秒 {question}")

    print("\n## 計測結果（全件）\n")
    print(results_table(results))
    print("\n## まとめ（成功した回の平均）\n")
    print(summary_table(summarize(results)))


if __name__ == "__main__":
    main()
