# Supplier extraction (SEC filings) and disruption news (GDELT)

**Problem.** Ownership and sanctions data say who a company *is*; they say nothing about whom it
*depends on*. That lives in prose: annual reports name key suppliers ("We utilize foundries,
such as Taiwan Semiconductor Manufacturing Company Limited, or TSMC…"), and news reports
disruptions. Milestone 4 turns both into typed, auditable graph links.

## Pipeline

```
seeds/sec_filers.csv ─► SEC EDGAR: latest 10-K / 20-F per filer (immutable, downloaded once)
        │
        ▼
 html → text → sentences ─► aliases learned from definitions ("…Limited, or TSMC")
        │                     └ merged across all filings (Intel says "TSMC" but never defines it)
        ▼
 candidate passages: a sentence with a supply cue AND a company name or known alias,
                     plus 3 sentences of preceding context ("Our competitors include:")
        ▼
 extractor ──► RULES (baseline)          ──► POSSIBLE_SUPPLIER  (review queue)
          └──► LLM (Claude, structured)  ──► SUPPLIES
        │      every mention must quote a sentence that occurs verbatim in the filing
        ▼
 name → LEI via GLEIF full-text search + strict name verification, then crawl the supplier's
        parents so the supplier joins the ownership graph
```

### Why two link types

The link type follows the method's measured precision (ontology v0.3). The rule baseline
finds 70 % of supplier relationships but only 56 % of what it finds is right, so its output is
a lead (`POSSIBLE_SUPPLIER`), never an assertion. Claude classifies *every* company in a
passage by its relationship to the filer, and its output is grounded (a hallucinated name or
quote is dropped), so it produces `SUPPLIES`.

## Evaluation

`supplygraph evaluate-extraction [--llm]` on the committed set in `eval/extraction/`:
82 candidate passages from the latest annual reports of 12 filers (Apple, NVIDIA, Intel, TSMC,
ASML, Walmart, Amazon, Unilever, AMD, Qualcomm, Broadcom, Tesla), hand-labelled by reading
every passage: **27 supplier relationships** (8 filers) + 4 marked ambiguous. Apple, Walmart,
Amazon and Unilever name no suppliers in their passages, which is itself realistic.

Unit = (filer, supplier entity), i.e. a graph edge. Labels record what the passage *states*:
Intel → TSMC is gold because Intel writes "we expect to continue to use TSMC and other
third-party foundry suppliers" — in the same section that calls TSMC "a key foundry
competitor".

| System | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Rules baseline (first run) | 0.244 | 0.704 | 0.362 | 19 | 59 | 8 |
| Rules baseline (after two generic parsing fixes) | **0.559** | **0.704** | **0.623** | 19 | 15 | 8 |
| Claude extractor | not yet measured — needs Anthropic credentials (`evaluate-extraction --llm`) | | | | | |

**Read the rule numbers as optimistic.** Its two fixes (section headings glued to "The Company"
parsed as company names; flattened subsidiary tables parsed as sentences) were found by reading
this set's false positives, and with 27 relationships there is no held-out split. They are
generic bugs, not tuned rules — but the honest comparison for the LLM is on new filings.

What the baseline cannot do (the remaining 15 false positives, by kind):

| Kind | Example |
|---|---|
| Customer, not supplier | AMD's "product purchase agreement with OpenAI"; TSMC's award *to* ASML ("TSMC Supplier Excellence Award") |
| Partner / JV | Intel's "strategic collaboration with UMC"; NXP and EDB in TSMC's SSMC joint venture |
| Capability, not supply | Intel: "few foundries capable … currently only TSMC and Samsung" |
| Software dependency | AMD and NVIDIA on Microsoft |
| Not a company | "Works Council of ASML Netherlands B.V.", "Euronext Amsterdam" (glossary) |

…and its 8 misses: names without a corporate suffix (Fabrinet, Panasonic, Foxconn Technology
Group, Qualcomm's "GlobalFoundries" and "Micron Technology"), and sentences without a supply
keyword ("we utilize TSMC, UMC and Samsung … for the production of our ICs").

### Entity linking (name → LEI), checked on the live run

GLEIF's full-text search returns loose candidates; acceptance is strict. Bugs found on real
names, each now a regression test:

| Name in filing | Naive result | Why | Fix |
|---|---|---|---|
| Samsung Electronics Co., Ltd. | ambiguous with the *Greek* subsidiary | Greek words vanish in normalization → "samsung electronics" | ignore name variants in scripts the normalizer can't read |
| GLOBALFOUNDRIES Inc. | ambiguous with GLOBALFOUNDRIES **U.S.** Inc. | single letters "u s" dropped | compare full names incl. legal form |
| Contemporary Amperex Technology Co. Limited (CATL) | Contemporary Amperex Technology **AG** (German subsidiary) | same words, different legal entity | a stated legal form must be compatible → left unresolved |
| Microsoft Corporation | a Belgian entity named "Microsoft" | normalized query only | search both the normalized and the written name |

Live result: 29 supplier names, 23 resolved; the 6 unresolved are CATL (the Chinese parent
isn't findable by its English name in GLEIF) and five non-suppliers.

### Known limitations

- **Candidate-filter recall is not measured.** Gold labels come from candidate passages, so a
  supplier named only in a sentence the filter skips is invisible to both extractor and gold
  (e.g. AMD: "SPIL and KYEC provide ATMP services" — no supply keyword).
- **Small gold set** (27 relationships, mostly semiconductors). Large-cap 10-Ks rarely name
  suppliers; adding automotive and pharma filers would broaden it.
- **One filing per company** (the latest). Supplier lists change year to year; history is a
  natural next step.

## Disruption news (GDELT)

GDELT publishes a Global Knowledge Graph file every 15 minutes with organizations and coded
themes already extracted. We read the raw files (the DOC search API rate-limits shared IPs
into uselessness), keep articles with a disruption theme from an explicit allow-list
(`NATURAL_DISASTER*`, `STRIKE`, `SHORTAGE`, `SANCTIONS`, `CYBER_ATTACK`, `DISASTER_FIRE`,
`MANMADE_DISASTER`, `POWER_OUTAGE`, `BLOCKADE`, `ECON_BANKRUPTCY`), and match their
organizations to **watchlist** companies (seeds, filers, and both ends of asserted `SUPPLIES`
links) by exact normalized name or filing-defined alias.

What a manual check of the live links showed, and what changed:

| | Company links (24 h) | Relevant disruption signals (judged from headlines) |
|---|---:|---|
| First version: article-level themes | 221 | ~4 in a sample of 40 (≈ 1 in 10) |
| + organization within 400 chars of a disruption theme, syndicated copies merged, watchlist from asserted links only | 14 | 5 of 14 |

The entity side was fine — "nvidia" really is NVIDIA. The noise was relevance: GDELT tags
themes per *article*, so a market wrap with one "storm" metaphor became a natural disaster
"mentioning" NVIDIA; one story syndicated 20 times became 20 events; and the rule baseline's
false suppliers (Microsoft) put unrelated companies under monitoring. Even after the fixes,
`MENTIONED_IN` is a **signal, not evidence** — the ontology says so, and nothing downstream
treats it as an assertion. The next step is an LLM relevance classifier with its own labelled
set, not more theme rules.
