// ============================================================================
// ai-engine/environment/respiratoryRisk.ts
//
// Converts environmental AQI information into a NON-DIAGNOSTIC
// respiratory-risk indicator.
//
// Inputs:
//   - AQI
//   - activity state
//   - optional vulnerability flag
//
// IMPORTANT:
//   This does NOT diagnose asthma, COPD, infection, hypoxia, etc.
//   It only indicates increased environmental respiratory risk.
//
// NOTE:
//   types.ts currently has no RespiratoryRiskResult contract.
//   This local interface is therefore intentionally isolated here.
//   When fusion.ts is implemented, we should formalize this contract
//   in types.ts if required by the final public API.
//
// Pure TypeScript.
// ============================================================================

import type {
  ActivityState,
  RiskLevel,
} from "../types";

export interface RespiratoryRiskResult {
  score: number | null;
  confidence: number;
  level: RiskLevel;
  evidence: string[];
}

type AQIBand =
  | "GOOD"
  | "SATISFACTORY"
  | "MODERATE"
  | "POOR"
  | "VERY_POOR"
  | "SEVERE";

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

/**
 * CPCB AQI category.
 *
 * CPCB bands:
 *   0–50       Good
 *   51–100     Satisfactory
 *   101–200    Moderate
 *   201–300    Poor
 *   301–400    Very Poor
 *   401–500    Severe
 */
function getAQIBand(aqi: number): AQIBand {
  if (aqi <= 50) {
    return "GOOD";
  }

  if (aqi <= 100) {
    return "SATISFACTORY";
  }

  if (aqi <= 200) {
    return "MODERATE";
  }

  if (aqi <= 300) {
    return "POOR";
  }

  if (aqi <= 400) {
    return "VERY_POOR";
  }

  return "SEVERE";
}

function baseRiskScore(
  band: AQIBand,
): number {
  switch (band) {
    case "GOOD":
      return 0.05;

    case "SATISFACTORY":
      return 0.15;

    case "MODERATE":
      return 0.35;

    case "POOR":
      return 0.55;

    case "VERY_POOR":
      return 0.75;

    case "SEVERE":
      return 0.90;
  }
}

/**
 * Convert numerical score to the engine's common risk levels.
 *
 * WHY:
 * These thresholds are intentionally broad and conservative.
 * The final alert decision is NOT made here; it still goes through
 * fusion + baseline + false-alarm gating.
 */
function scoreToRiskLevel(
  score: number,
): RiskLevel {
  if (score >= 0.70) {
    return "RISK";
  }

  if (score >= 0.35) {
    return "CAUTION";
  }

  return "NORMAL";
}

/**
 * Activity modifier.
 *
 * WHY:
 * Increased exertion can increase respiratory exposure because
 * ventilation generally rises with activity. We therefore use
 * activity as a risk amplifier rather than treating AQI alone
 * as the complete context.
 */
function activityMultiplier(
  activityState: ActivityState,
): number {
  switch (activityState) {
    case "ACTIVE":
      return 1.20;

    case "LIGHT":
      return 1.08;

    case "REST":
      return 1.00;

    default:
      return 1.00;
  }
}

/**
 * Calculate respiratory environmental risk.
 *
 * `vulnerabilityFlag` is intentionally optional because the current
 * SensorTickInput does not contain a vulnerability profile.
 *
 * Future user-profile integration can pass this value without changing
 * the environmental AQI calculation itself.
 */
export function calculateRespiratoryRisk(
  aqi: number | null,
  activityState: ActivityState,
  vulnerabilityFlag = false,
  aqiConfidence = 1,
): RespiratoryRiskResult {
  if (
    aqi === null ||
    !Number.isFinite(aqi) ||
    aqi < 0
  ) {
    return {
      score: null,
      confidence: 0,
      level: "NORMAL",
      evidence: [
        "Respiratory risk unavailable because AQI data is unavailable.",
      ],
    };
  }

  /**
   * AQI values above 500 are not accepted by this Phase-1 interface.
   */
  if (aqi > 500) {
    return {
      score: null,
      confidence: 0,
      level: "NORMAL",
      evidence: [
        "Respiratory risk unavailable because AQI is outside the supported range.",
      ],
    };
  }

  const band = getAQIBand(aqi);

  let score = baseRiskScore(band);

  const multiplier =
    activityMultiplier(activityState);

  score *= multiplier;

  /**
   * Vulnerability is an amplifier, not a separate diagnosis.
   *
   * WHY:
   * The CPCB descriptions explicitly note greater potential impact
   * in sensitive groups at higher AQI levels.
   */
  if (vulnerabilityFlag) {
    score += 0.10;
  }

  score = clamp01(score);

  const evidence: string[] = [
    `AQI falls in the ${band} environmental band.`,
  ];

  if (activityState === "ACTIVE") {
    evidence.push(
      "Active movement increases the environmental respiratory-risk context.",
    );
  } else if (activityState === "LIGHT") {
    evidence.push(
      "Light activity modestly increases the environmental respiratory-risk context.",
    );
  } else {
    evidence.push(
      "Resting activity does not add an exertion multiplier.",
    );
  }

  if (vulnerabilityFlag) {
    evidence.push(
      "User vulnerability profile increases the environmental risk weighting.",
    );
  }

  /**
   * Confidence cannot exceed the confidence of the AQI source.
   *
   * WHY:
   * A risk score calculated from an uncalibrated MQ135 proxy must
   * remain low-confidence even when the numerical proxy is high.
   */
  const confidence = clamp01(aqiConfidence);

  return {
    score,
    confidence,
    level: scoreToRiskLevel(score),
    evidence,
  };
}