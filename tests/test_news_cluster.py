from pathlib import Path

from supplygraph import news_cluster as nc

EVAL = Path(__file__).resolve().parent.parent / "eval" / "news"


def ev(i: str, title: str | None, ts: str = "2026-11-07T10:00:00+00:00", **kw) -> dict:
    return {"id": i, "title": title, "published_at": ts, "organizations": [],
            "event_types": [], "copies": 1, **kw}


def test_rewritten_headlines_of_one_story_merge() -> None:
    events = [
        ev("a", "5.7-magnitude quake hits central Japan, 20 injured"),
        ev("b", "Magnitude 5.7 earthquake hits central Japan, dozens injured"),
        ev("c", "Lando Norris claims pole for Sao Paulo GP as rain disrupts qualifying"),
    ]
    assert nc.cluster_stories(events) == [[0, 1], [2]]


def test_one_shared_word_never_merges() -> None:
    # "Iran" alone is not a story, however rare the word is
    events = [ev("a", "Iran"), ev("b", "Iran wins")]
    assert nc.cluster_stories(events, threshold=0.1) == [[0], [1]]


def test_time_window_separates_recurring_headlines() -> None:
    a = ev("a", "AP News Summary at 9:54 p.m. EST", "2026-11-05T01:00:00+00:00")
    b = ev("b", "AP News Summary at 9:54 p.m. EST", "2026-11-08T01:00:00+00:00")
    assert nc.cluster_stories([a, b]) == [[0], [1]]


def test_untitled_and_empty_inputs_never_merge() -> None:
    assert nc.cluster_stories([]) == []
    assert nc.cluster_stories([ev("a", None), ev("b", None), ev("c", "")]) == [[0], [1], [2]]


def test_merge_keeps_earliest_and_unions_metadata() -> None:
    a = ev("a", "Oil facility explosion in Assam, death toll rises to three",
           "2026-11-07T12:00:00+00:00", organizations=["oil india"], event_types=["FIRE"],
           copies=2)
    b = ev("b", "Death toll in explosion at Oil India facility in Assam rises to three",
           "2026-11-07T09:00:00+00:00", organizations=["ongc"], event_types=["INDUSTRIAL_ACCIDENT"])
    (out,) = nc.merge_stories([a, b])
    assert out["id"] == "b" and out["copies"] == 3
    assert out["organizations"] == ["ongc", "oil india"]
    assert out["event_types"] == ["FIRE", "INDUSTRIAL_ACCIDENT"]


def test_merge_does_not_mutate_inputs() -> None:
    a, b = ev("a", "Storm floods tents in Gaza"), ev("b", "Gaza storm floods tents")
    nc.merge_stories([a, b])
    assert a["copies"] == 1 and b["copies"] == 1


def test_different_places_same_template_stay_apart() -> None:
    # regression: dev sweep showed 0.4 merged "earthquake strikes Namibia" with "...Japan";
    # the production threshold must keep them apart
    events = [ev("a", "Magnitude-5.1 earthquake strikes northwestern Namibia"),
              ev("b", "A strong earthquake with a preliminary magnitude of 5.7 strikes Japan")]
    assert nc.cluster_stories(events, nc.DEFAULT_THRESHOLD) == [[0], [1]]


def test_score_pairs_edge_cases() -> None:
    assert nc.score_pairs(set(), {}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                         "tp": 0, "fp": 0, "fn": 0}
    m = nc.score_pairs({(0, 1), (2, 3)}, {(0, 1): True, (2, 3): False, (4, 5): True})
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)


def test_committed_eval_set_pins_the_numbers() -> None:
    # docs/news-clustering.md quotes these; update it when they move
    r = nc.evaluate(EVAL / "events.jsonl", EVAL / "gold_pairs.csv")
    tfidf = r["systems"]["tfidf_single_linkage"]
    assert tfidf["threshold"] == nc.DEFAULT_THRESHOLD
    assert (tfidf["test"]["tp"], tfidf["test"]["fp"], tfidf["test"]["fn"]) == (27, 3, 10)
    assert r["systems"]["exact_title"]["test"]["tp"] == 0  # the pool is what exact dedupe missed
    assert tfidf["test"]["recall"] > r["systems"]["jaccard"]["test"]["recall"]
