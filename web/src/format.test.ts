import { describe, expect, it } from "vitest";
import { RISK_RAMP, fmtScore, riskColor, splitCitations } from "./format";

describe("riskColor", () => {
  it("zero and missing risk recede to the surface token", () => {
    expect(riskColor(0)).toBe("var(--risk-0)");
    expect(riskColor(Number.NaN)).toBe("var(--risk-0)");
  });
  it("steps up the ramp monotonically and saturates at 1", () => {
    const steps = [0.01, 0.07, 0.2, 0.4, 0.6, 1.0].map((s) => RISK_RAMP.indexOf(riskColor(s)));
    expect(steps).toEqual([0, 1, 2, 3, 4, 5]);
    expect(riskColor(1.5)).toBe(RISK_RAMP[5]);
  });
});

describe("fmtScore", () => {
  it("formats edge cases", () => {
    expect(fmtScore(null)).toBe("–");
    expect(fmtScore(0)).toBe("0");
    expect(fmtScore(0.004)).toBe("<0.01");
    expect(fmtScore(0.81368)).toBe("0.81");
  });
});

describe("splitCitations", () => {
  it("extracts refs, including several in one bracket, and keeps other brackets as text", () => {
    const parts = splitCitations("TSMC [lei:A] supplies [lei:B, filing:0001-26-1] (see [note]).");
    expect(parts).toEqual([
      "TSMC ",
      { ref: "lei:A" },
      " supplies ",
      { ref: "lei:B" },
      { ref: "filing:0001-26-1" },
      " (see [note]).",
    ]);
  });
});
