"""計測スクリプトのテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

from google.genai import errors

from app import bench


class FakeModels:
    def __init__(self, usage=None, error=None):
        self.usage = usage
        self.error = error

    def generate_content(self, model, contents):
        if self.error:
            raise self.error
        return SimpleNamespace(usage_metadata=self.usage, text="答え")


def fake_client(**kwargs):
    return SimpleNamespace(models=FakeModels(**kwargs))


def test_measure_records_tokens():
    usage = SimpleNamespace(
        prompt_token_count=7, candidates_token_count=600, thoughts_token_count=None, total_token_count=607
    )
    r = bench.measure(fake_client(usage=usage), "m", "q", 1)
    assert r.status == "ok"
    assert (r.prompt_tokens, r.output_tokens, r.thoughts_tokens, r.total_tokens) == (7, 600, None, 607)
    assert r.seconds >= 0


def test_measure_without_usage_metadata():
    r = bench.measure(fake_client(usage=None), "m", "q", 1)
    assert r.status == "ok"
    assert r.output_tokens is None


def test_measure_records_api_error_instead_of_raising():
    error = errors.ClientError(429, {"error": {"message": "rate limit"}})
    r = bench.measure(fake_client(error=error), "m", "q", 1)
    assert r.status == "429"
    assert r.output_tokens is None


def test_summarize_averages_only_successful_runs():
    results = [
        bench.Result("m", "q", 1, "ok", 1.0, 10, 100, 0, 110),
        bench.Result("m", "q", 2, "ok", 3.0, 10, 300, 0, 310),
        bench.Result("m", "q", 3, "429", 0.1),
    ]
    [row] = bench.summarize(results)
    assert row["ok"] == 2
    assert row["errors"] == ["429"]
    assert row["mean_sec"] == 2.0
    assert row["mean_output"] == 200
    assert row["output_per_sec"] == 100


def test_summarize_when_all_runs_failed():
    [row] = bench.summarize([bench.Result("m", "q", 1, "503", 0.5)])
    assert row["ok"] == 0
    assert row["mean_sec"] is None
    assert row["output_per_sec"] is None
    assert "| 0 | 503 | - |" in bench.summary_table([row])


def test_summarize_empty():
    assert bench.summarize([]) == []


def test_tables_render_missing_values_as_dash():
    r = bench.Result("m", "q", 1, "ok", 1.234)
    assert "| 1.23 | - | - | - | - |" in bench.results_table([r])
