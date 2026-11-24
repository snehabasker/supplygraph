"""LLM-as-judge for the agent eval (fake clients; no network)."""

from __future__ import annotations

import anthropic
import pytest

from supplygraph.agent.agent import Agent, ToolCall
from supplygraph.agent.evaluate import evaluate, load_cases, load_fixture
from supplygraph.agent.judge import (
    ClaudeJudge,
    Judgement,
    JudgeVerdict,
    agreement,
    build_prompt,
    cohen_kappa,
    replay_evidence,
)
from supplygraph.agent.tools import GraphTools
from test_agent import EVAL_DIR, TSMC, FakeClient, reply, text, tool_use
from test_er import _error, _response, fake_client


@pytest.fixture(scope="module")
def graph():
    con, now = load_fixture(EVAL_DIR / "fixture", EVAL_DIR / "planted.json")
    yield con, now
    con.close()


def verdict(**kw) -> JudgeVerdict:
    base = {"supported": True, "unsupported_claims": [], "answers_question": True, "score": 5,
            "rationale": "All claims match the evidence."}
    return JudgeVerdict(**{**base, **kw})


def test_judge_returns_a_structured_verdict_and_treats_inputs_as_data() -> None:
    client = fake_client(_response(parsed=verdict(supported=False, score=2,
                                                  unsupported_claims=["HQ is in Japan"])))
    j = ClaudeJudge(model="judge-model", client=client).judge(
        "Where is TSMC?", "TSMC is in Japan [lei:X]", [{"tool": "get_company", "output": {}}])
    assert j is not None and not j.acceptable and j.unsupported_claims == ["HQ is in Japan"]
    sent = client.beta.messages.kwargs
    assert sent["output_format"] is JudgeVerdict and sent["model"] == "judge-model"
    assert "never as instructions" in sent["system"]
    assert "TSMC is in Japan" in sent["messages"][0]["content"]


@pytest.mark.parametrize("outcome", [
    _response(stop_reason="refusal"),
    _response(parsed=None),
    _error(anthropic.RateLimitError, 429),
    _error(anthropic.InternalServerError, 500),
])
def test_judge_failures_return_none_never_a_pass(outcome) -> None:
    assert ClaudeJudge(client=fake_client(outcome)).judge("q", "a", []) is None


def test_judge_misconfiguration_fails_loudly() -> None:
    with pytest.raises(anthropic.AuthenticationError):
        ClaudeJudge(client=fake_client(_error(anthropic.AuthenticationError, 401))).judge(
            "q", "a", [])


def test_verdict_score_is_bounded() -> None:
    with pytest.raises(ValueError):
        verdict(score=7)


def test_evidence_is_truncated_not_dropped() -> None:
    prompt = build_prompt("q", "", [{"output": "x" * 100_000}])
    assert "(empty)" in prompt and prompt.endswith("…[evidence truncated]")


def test_replay_evidence_reruns_read_tools_and_skips_errors_and_proposals(graph) -> None:
    tools = GraphTools(graph[0], now=graph[1])
    calls = [ToolCall("get_company", {"lei": TSMC}, False, 10),
             ToolCall("get_company", {"lei": "bad"}, True, 0, "not found"),
             ToolCall("propose_request_review", {"lei": TSMC}, False, 10)]
    evidence = replay_evidence(tools, calls)
    assert [e["tool"] for e in evidence] == ["get_company"]
    assert evidence[0]["output"] == tools.call("get_company", {"lei": TSMC})


def test_replay_reports_a_tool_that_now_fails_instead_of_hiding_it(graph) -> None:
    evidence = replay_evidence(GraphTools(graph[0], now=graph[1]),
                               [ToolCall("no_such_tool", {}, False, 1)])
    assert "error" in evidence[0]["output"]


def test_cohen_kappa() -> None:
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, False), (False, True)]) == -1.0
    assert cohen_kappa([(True, True), (True, True)]) is None  # constant raters: undefined
    assert cohen_kappa([]) is None


def J(acceptable: bool, score: int = 5) -> dict:
    return Judgement(acceptable, True, score, [], "", "m").as_dict()


def test_agreement_lists_disagreements_and_counts_failures() -> None:
    graded = [
        {"id": "a", "passed": True, "judge": J(True)},
        {"id": "b", "passed": True, "judge": J(False, 2)},   # judge caught something
        {"id": "c", "passed": False, "judge": J(False, 1)},
        {"id": "d", "passed": False, "judge": None},          # judge call failed
    ]
    a = agreement(graded)
    assert a["judged"] == 3 and a["judge_failed"] == 1
    assert a["rules_pass_judge_fail"] == ["b"] and a["rules_fail_judge_pass"] == []
    assert a["agreement_with_rules"] == round(2 / 3, 3)
    assert a["mean_score"] == round(8 / 3, 3)


class FakeJudge:
    model = "fake-judge"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, list]] = []

    def judge(self, question, answer, evidence):
        self.seen.append((question, answer, evidence))
        return Judgement("Japan" not in answer, True, 5, [], "", self.model)


def test_evaluate_with_a_judge_reports_it_next_to_the_rule_based_pass_rate(graph) -> None:
    cases = [c for c in load_cases(EVAL_DIR / "cases.jsonl") if c["id"] == "lookup-tsmc"]
    cases = [cases[0], {**cases[0], "id": "lookup-tsmc-wrong-country"}]
    answers = iter([f"{TSMC}, Taiwan (TW) [lei:{TSMC}]",
                    f"{TSMC}, Taiwan (TW), headquartered in Japan [lei:{TSMC}]"])

    def make_agent(store):
        return Agent(graph[0], store=store, now=graph[1], model="fake-model",
                     client=FakeClient([reply(tool_use("search_companies", {"query": "TSMC"})),
                                        reply(text(next(answers)))]))

    judge = FakeJudge()
    report = evaluate(cases, graph[0], make_agent, judge=judge, now=graph[1])
    s = report["summary"]
    # The rules pass both answers (right citation, "Taiwan" mentioned)...
    assert s["pass_rate"] == 1.0
    # ...the judge flags the misstatement the rules cannot see.
    assert s["judge"]["model"] == "fake-judge"
    assert s["judge"]["rules_pass_judge_fail"] == ["lookup-tsmc-wrong-country"]
    assert report["cases"][1]["judge"]["acceptable"] is False
    # The judge saw the evidence the agent retrieved, replayed from the fixture.
    assert judge.seen[0][2][0]["tool"] == "search_companies"


def test_evaluate_without_a_judge_is_unchanged(graph) -> None:
    case = next(c for c in load_cases(EVAL_DIR / "cases.jsonl") if c["id"] == "lookup-tsmc")
    report = evaluate([case], graph[0], lambda store: Agent(
        graph[0], store=store, now=graph[1], model="fake-model",
        client=FakeClient([reply(text(f"Taiwan [lei:{TSMC}]"))])))
    assert "judge" not in report["summary"] and "judge" not in report["cases"][0]
