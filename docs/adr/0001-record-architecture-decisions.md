# 0001. Record architecture decisions

**Status:** Accepted

## Context
SupplyGraph was built in milestones, and several decisions were made deliberately against an
obvious default: no GNN, no vector index, an agent that cannot act. Without a record, those
choices look like omissions, and get "fixed" by someone who never saw the measurements behind
them.

## Decision
Every decision that constrains future work gets a short record in `docs/adr/`:
- **context** (the forces at play);
- **decision**;
- **alternatives considered**;
- **consequences**, good and bad.

Measurements live in the feature docs (`docs/*.md`), and records link to them rather than copy
them. Records are append-only: reversing a decision means writing a new record that supersedes
the old one.

## Consequences
Records 0002–0009 were written retroactively at Milestone 8, from the design docs and commit
history of Milestones 1–7. Records 0010–0012 were written with the decisions.
