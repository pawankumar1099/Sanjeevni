// ============================================================================
// ai-engine/decision/riskDecisionEngine.ts
//
// Final Phase-1 risk decision layer.
//
// Responsibilities:
//   - Convert a candidate score into NORMAL / CAUTION / RISK.
//   - Respect the False-Alarm Gate.
//   - Produce calm, non-diagnostic recommendations.
//   - Determine whether SOS should be recommended.
//
// This module does NOT:
//   - process raw ECG
//   - detect R-peaks
//   - calculate HRV
//   - detect falls
//   - access React Native
//   - send an SOS itself
//
// The mobile layer owns actual notification/SOS execution.
//
// Phase 1 uses three risk levels because the architecture deliberately avoids
// arbitrary five-level clinical-looking thresholds without validation data.
// ============================================================================

import type {
  CategoryRiskDecision,
  FalseAlarmGateOutput,
  RiskLevel,
} from "../types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface RiskDecisionEngineOptions {
  /**
   * Score at/above this level becomes RISK if the false-alarm gate passes.
   */
  riskScoreThreshold?: number;

  /**
   * Score at/above this level becomes CAUTION.
   */
  cautionScoreThreshold?: number;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
//
// These are prototype decision boundaries, NOT medical thresholds.
//
// The architecture specifies three qualitative states, but does not provide
// clinically validated numerical cutoffs. Therefore these values are kept
// simple and deterministic for Phase 1 and must be validated later.
// ---------------------------------------------------------------------------

const DEFAULT_CAUTION_SCORE_THRESHOLD = 0.45;

const DEFAULT_RISK_SCORE_THRESHOLD = 0.70;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

type DecisionCategory =
  | "cardiac"
  | "heat"
  | "respiratory"
  | "fall";

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

function clamp01(
  value: number,
): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(
    0,
    Math.min(1, value),
  );
}

// ---------------------------------------------------------------------------
// Risk level
// ---------------------------------------------------------------------------

function scoreToLevel(
  score: number,
  gatePassed: boolean,
  cautionThreshold: number,
  riskThreshold: number,
): RiskLevel {
  // No candidate can become an alert level until the False-Alarm Gate passes.
  //
  // This is important because the decision engine must never bypass:
  // quality → motion → persistence → baseline → confidence → cooldown.
  if (!gatePassed) {
    return "NORMAL";
  }

  if (
    score >= riskThreshold
  ) {
    return "RISK";
  }

  if (
    score >= cautionThreshold
  ) {
    return "CAUTION";
  }

  return "NORMAL";
}

// ---------------------------------------------------------------------------
// Recommendations
// ---------------------------------------------------------------------------

function recommendationFor(
  category: DecisionCategory,
  level: RiskLevel,
): string {
  if (level === "NORMAL") {
    return "No immediate action indicated by the current sensor-derived risk assessment.";
  }

  if (
    category === "cardiac"
  ) {
    if (level === "RISK") {
      return "Pause strenuous activity and seek appropriate medical assistance if symptoms are present or the concern persists.";
    }

    return "Consider resting and monitoring how you feel. Recheck the signal if the concern continues.";
  }

  if (
    category === "heat"
  ) {
    if (level === "RISK") {
      return "Move to a cooler or shaded area, reduce exertion, and hydrate appropriately.";
    }

    return "Consider reducing exertion, seeking shade, and maintaining hydration.";
  }

  if (
    category === "respiratory"
  ) {
    if (level === "RISK") {
      return "Reduce exposure to polluted air and move to cleaner air if possible, especially if breathing discomfort is present.";
    }

    return "Consider reducing prolonged outdoor exertion and moving to cleaner air if practical.";
  }

  // Fall
  if (level === "RISK") {
    return "Check for injury and remain still if needed. Contact an emergency contact or appropriate assistance if help is required.";
  }

  return "Check your surroundings and condition after the detected movement event.";
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

function appendUnique(
  evidence: string[],
  message: string,
): void {
  if (
    !evidence.includes(message)
  ) {
    evidence.push(message);
  }
}

// ---------------------------------------------------------------------------
// Risk Decision Engine
// ---------------------------------------------------------------------------

export class RiskDecisionEngine {
  private readonly cautionScoreThreshold: number;

  private readonly riskScoreThreshold: number;

  constructor(
    options: RiskDecisionEngineOptions = {},
  ) {
    this.cautionScoreThreshold =
      clamp01(
        options.cautionScoreThreshold ??
          DEFAULT_CAUTION_SCORE_THRESHOLD,
      );

    this.riskScoreThreshold =
      clamp01(
        options.riskScoreThreshold ??
          DEFAULT_RISK_SCORE_THRESHOLD,
      );

    // A risk threshold lower than caution would make the decision ordering
    // mathematically inconsistent. Instead of throwing at runtime, we repair
    // it deterministically by keeping RISK above CAUTION.
    if (
      this.riskScoreThreshold <
      this.cautionScoreThreshold
    ) {
      this.riskScoreThreshold =
        Math.min(
          1,
          this.cautionScoreThreshold +
            0.10,
        );
    }
  }

  // -------------------------------------------------------------------------
  // Generic category decision
  // -------------------------------------------------------------------------

  /**
   * Produces the final decision for cardiac/heat/respiratory.
   *
   * The candidate score and confidence are supplied separately so this method
   * remains compatible with the existing FusionOutput and Gate contracts.
   */
  decide(
    category:
      | "cardiac"
      | "heat"
      | "respiratory"
      | "fall",
    score: number,
    confidence: number,
    gate: FalseAlarmGateOutput,
    timestamp: number,
    evidence: string[] = [],
  ): CategoryRiskDecision {
    const safeScore =
      clamp01(score);

    const safeConfidence =
      clamp01(confidence);

    const finalEvidence =
      [...evidence];

    // -----------------------------------------------------------------------
    // Gate result is authoritative.
    // -----------------------------------------------------------------------

    if (
      gate.shouldAlert
    ) {
      appendUnique(
        finalEvidence,
        "False-Alarm Gate: PASSED",
      );
    } else if (
      gate.suppressedReason
    ) {
      appendUnique(
        finalEvidence,
        `False-Alarm Gate: suppressed (${gate.suppressedReason})`,
      );
    } else {
      appendUnique(
        finalEvidence,
        "False-Alarm Gate: candidate not promoted",
      );
    }

    for (
      const item of gate.evidence
    ) {
      appendUnique(
        finalEvidence,
        item,
      );
    }

    const level =
      scoreToLevel(
        safeScore,
        gate.shouldAlert,
        this.cautionScoreThreshold,
        this.riskScoreThreshold,
      );

    // -----------------------------------------------------------------------
    // Evidence about the resulting level.
    // -----------------------------------------------------------------------

    if (
      level === "RISK"
    ) {
      appendUnique(
        finalEvidence,
        `Risk score ${safeScore.toFixed(2)} reached RISK threshold`,
      );
    } else if (
      level === "CAUTION"
    ) {
      appendUnique(
        finalEvidence,
        `Risk score ${safeScore.toFixed(2)} reached CAUTION range`,
      );
    } else {
      appendUnique(
        finalEvidence,
        "Current candidate does not produce an alert-level decision",
      );
    }

    // -----------------------------------------------------------------------
    // SOS policy
    // -----------------------------------------------------------------------
    //
    // SOS is deliberately NOT recommended for every RISK category.
    //
    // Phase-1 architecture specifically identifies sustained RISK-level
    // cardiac/fall situations as the pathway for the mobile team's SOS logic.
    //
    // This engine only sets a flag. It never sends the SOS itself.
    // -----------------------------------------------------------------------

    const sosRecommended =
      level === "RISK" &&
      (
        category === "cardiac" ||
        category === "fall"
      );

    if (
      sosRecommended
    ) {
      appendUnique(
        finalEvidence,
        "Sustained high-priority risk may warrant SOS handling",
      );
    }

    return {
      category,
      level,
      score: safeScore,
      confidence: safeConfidence,
      evidence: finalEvidence,
      timestamp,
      recommendedAction:
        recommendationFor(
          category,
          level,
        ),
      sosRecommended,
    };
  }

  // -------------------------------------------------------------------------
  // Category-specific helpers
  // -------------------------------------------------------------------------

  decideCardiac(
    score: number,
    confidence: number,
    gate: FalseAlarmGateOutput,
    timestamp: number,
    evidence: string[] = [],
  ): CategoryRiskDecision {
    return this.decide(
      "cardiac",
      score,
      confidence,
      gate,
      timestamp,
      evidence,
    );
  }

  decideHeat(
    score: number,
    confidence: number,
    gate: FalseAlarmGateOutput,
    timestamp: number,
    evidence: string[] = [],
  ): CategoryRiskDecision {
    return this.decide(
      "heat",
      score,
      confidence,
      gate,
      timestamp,
      evidence,
    );
  }

  decideRespiratory(
    score: number,
    confidence: number,
    gate: FalseAlarmGateOutput,
    timestamp: number,
    evidence: string[] = [],
  ): CategoryRiskDecision {
    return this.decide(
      "respiratory",
      score,
      confidence,
      gate,
      timestamp,
      evidence,
    );
  }

  decideFall(
    score: number,
    confidence: number,
    gate: FalseAlarmGateOutput,
    timestamp: number,
    evidence: string[] = [],
  ): CategoryRiskDecision {
    return this.decide(
      "fall",
      score,
      confidence,
      gate,
      timestamp,
      evidence,
    );
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createRiskDecisionEngine(
  options: RiskDecisionEngineOptions = {},
): RiskDecisionEngine {
  return new RiskDecisionEngine(
    options,
  );
}