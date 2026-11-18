from app import retrieval
from app.retrieval import add_neighbor_pages


def page(document_id: str, n: int) -> dict:
    return {"page_id": f"{document_id}-p{n:03d}", "document_id": document_id, "page": n, "content": f"本文{n}"}


PAGES = [page("lec01-main", n) for n in (1, 2, 3)] + [page("lec02-main", n) for n in (1, 2, 3, 4, 5)]


def ids(results: list[dict]) -> list[str]:
    return [r["page_id"] for r in results]


def test_window_zero_returns_hits_unchanged():
    hits = [PAGES[4]]
    assert add_neighbor_pages(hits, PAGES, 0) is hits


def test_adds_previous_and_next_page_after_each_hit():
    results = add_neighbor_pages([PAGES[4]], PAGES, 1)
    assert ids(results) == ["lec02-main-p002", "lec02-main-p001", "lec02-main-p003"]
    assert results[0]["retrieved_by"] == "search"
    assert results[1]["retrieved_by"] == "neighbor"
    assert results[1]["neighbor_of"] == "lec02-main-p002"


def test_does_not_cross_documents():
    results = add_neighbor_pages([PAGES[2], PAGES[3]], PAGES, 1)
    assert ids(results) == ["lec01-main-p003", "lec01-main-p002", "lec02-main-p001", "lec02-main-p002"]


def test_shared_neighbor_and_hit_pages_are_not_duplicated():
    # p002 と p004 の検索結果は、どちらも p003 を隣に持つ。p004 自身は検索結果として1回だけ返す
    results = add_neighbor_pages([PAGES[4], PAGES[6], PAGES[5]], PAGES, 1)
    assert sorted(ids(results)) == sorted({p["page_id"] for p in PAGES[3:]})
    assert len(results) == len(set(ids(results)))
    assert results[-1] == {**PAGES[5], "retrieved_by": "search"}


def test_hit_outside_page_list_is_kept_without_neighbors():
    stray = page("lec09-main", 1)
    assert ids(add_neighbor_pages([stray], PAGES, 1)) == ["lec09-main-p001"]


def test_search_with_neighbors_uses_search_and_load_pages(monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda query: [PAGES[4]])
    monkeypatch.setattr(retrieval, "load_pages", lambda *_: PAGES)
    assert ids(retrieval.search_with_neighbors("q", 0)) == ["lec02-main-p002"]
    assert len(retrieval.search_with_neighbors("q", 1)) == 3
