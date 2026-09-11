// ============================================================================
// ai-engine/ecg/rhythmAnomaly.ts
// Multi-stage rhythm anomaly flag: personal-baseline z-score on RR interval
// behavior + temporal persistence, BEFORE anything is allowed to look like
// a flag. This module does NOT decide alerts — it only produces a
// RhythmFlag that riskDecisionEngine.ts (via the False-Alarm Gate) will
// further gate on quality/motion/cooldown (Rule 5).
//
// IMPORTANT (Rule 6): "detected" here means "a statistically anomalous
// beat-to-beat pattern was observed relative to this person's own recent
// history" — it is explicitly NOT a rhythm classification (not "AFib",
// not "PVC"). Evidence strings must stay in that same risk-oriented,
// non-diagnostic register.
// ============================================================================

import { RhythmFlag } from "../types";
import { zScore, mean, stdDev, clamp } from "../utils/math";

export interface RhythmAnomalyOptions {
  /** Personal baseline for RR interval variability (e.g. from
   * personalBaseline.ts, metric = "rrVariance" or similar). If baseline
   * history is insufficient (cold start), we deliberately cannot compute
   * a meaningful z-score and must not fabricate one (Rule 4). */
  baselineMeanRR: number | null;
  baselineStdDevRR: number | null;
  hasSufficientBaselineHistory: boolean;
  /** How many consecutive short windows have shown anomalous RR behavior
   * so far (persistence counter maintained by the caller across ticks) —
   * required before "detected" can become true (Rule 5: temporal
   * persistence, not a single anomalous beat). */
  consecutiveAnomalousWindows: number;
  /** Threshold of consecutive anomalous windows required to flag. */
  persistenceThreshold?: number;
  /** |z| beyond which a window's RR pattern counts as "anomalous" for the
   * persistence counter. 2.5 is a conventional "notably unusual, not yet
   * extreme" statistical cutoff. */
  zScoreThreshold?: number;
}

export function evaluateRhythmAnomaly(
  rrIntervalsMs: number[],
  options: RhythmAnomalyOptions
): RhythmFlag {
  const {
    baselineMeanRR,
    baselineStdDevRR,
    hasSufficientBaselineHistory,
    consecutiveAnomalousWindows,
    persistenceThreshold = 3,
    zScoreThreshold = 2.5,
  } = options;

  const evidence: string[] = [];

  // Not enough RR data this window to say anything at all.
  if (rrIntervalsMs.length < 5) {
    return { detected: false, confidence: 0, evidence: ["Insufficient RR data in current window"] };
  }

  // Cold-start guard: without a personal baseline we cannot claim anything
  // is "unusual for this person" — never fabricate a synthetic population
  // baseline to fake confidence here (Rule 4/6).
  if (!hasSufficientBaselineHistory || baselineMeanRR === null || baselineStdDevRR === null) {
    return {
      detected: false,
      confidence: 0,
      evidence: ["Personal baseline still building — anomaly detection not yet active"],
    };
  }

  // Compare this window's own mean RR against the personal baseline mean,
  // using the baseline's stdDev as the scale — answers "how unusual is
  // this window's average beat spacing for THIS person specifically".
  const windowMeanRR = mean(rrIntervalsMs);
  const windowStdDevRR = stdDev(rrIntervalsMs);
  if (windowMeanRR === null) {
    return { detected: false, confidence: 0, evidence: ["Unable to compute window RR mean"] };
  }

  const z = zScore(windowMeanRR, baselineMeanRR, baselineStdDevRR);
  if (z === null) {
    return { detected: false, confidence: 0, evidence: ["Baseline variance too low to score against"] };
  }

  const isWindowAnomalous = Math.abs(z) >= zScoreThreshold;

  if (isWindowAnomalous) {
    evidence.push(
      `Window mean RR interval deviates ${Math.abs(z).toFixed(1)} standard deviations from personal baseline`
    );
  }
  if (windowStdDevRR !== null && windowStdDevRR > baselineStdDevRR * 2) {
    evidence.push("Beat-to-beat variability substantially higher than personal baseline");
  }

  // Persistence gate: only "detected" once the CALLER's running persistence
  // counter (tracked across ticks, since this module is stateless per call)
  // has reached the threshold. This function reports whether THIS window
  // is anomalous; the caller is responsible for incrementing/resetting
  // consecutiveAnomalousWindows and re-invoking — keeping this module pure
  // and easy to swap for a Phase 2 model later (Rule 8).
  const detected = isWindowAnomalous && consecutiveAnomalousWindows >= persistenceThreshold;

  if (detected) {
    evidence.push(
      `Anomalous pattern persisted across ${consecutiveAnomalousWindows} consecutive windows`
    );
  } else if (isWindowAnomalous) {
    evidence.push("Anomalous window observed but not yet persistent — monitoring continues");
  }

  // Confidence blends how extreme the z-score is with how much persistence
  // has accumulated, capped at 1.0 — a single very extreme window still
  // won't outrank the persistence requirement for "detected" above, but it
  // does raise confidence once detected.
  const zConfidence = clamp((Math.abs(z) - zScoreThreshold) / zScoreThreshold, 0, 1);
  const persistenceConfidence = clamp(consecutiveAnomalousWindows / persistenceThreshold, 0, 1);
  const confidence = detected ? clamp(0.5 * zConfidence + 0.5 * persistenceConfidence, 0, 1) : 0;

  if (evidence.length === 0) {
    evidence.push("RR pattern within personal baseline range");
  }

  return { detected, confidence, evidence };
}