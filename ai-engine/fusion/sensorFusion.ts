// ============================================================================
// ai-engine/fusion/sensorFusion.ts
//
// Confidence-aware rule-based fusion for Sanjeevni.
//
// IMPORTANT:
// The current locked FusionInput contract contains ONE candidate value per
// category rather than an array of independent sensor contributions.
//
// Therefore, this file does NOT invent additional inputs or pretend that
// multiple measurements are being mathematically averaged.
//
// Instead it performs:
//   1. input validity checks
//   2. confidence normalization
//   3. ECG SQI-aware confidence adjustment
//   4. motion-aware confidence adjustment
//   5. category-specific evidence generation
//   6. insufficient-data protection
//   7. fall-result passthrough
//
// This preserves the current types.ts contract while leaving the module
// interface stable for a future richer fusion input.
//
// Phase 1: DSP/statistics/rules only. No ML/DL.
// ============================================================================

import type {
  ActivityState,
  FusedCategoryRisk,
  FusionInput,
  FusionOutput,
  RiskLevel,
  SQI,
} from "../types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
//
// These thresholds are intentionally conservative.
//
// Fusion is not supposed to manufacture a risk value when upstream data is
// unreliable. The decision engine later performs the final alert gating.
// Fusion's job is to represent how trustworthy the candidate evidence is.
//
// 0.30 is deliberately not a "clinical" threshold. It is an engineering
// minimum-data threshold: below this, the system should explicitly say
// "insufficient data" rather than present a weak number as meaningful.
// ---------------------------------------------------------------------------

const MIN_FUSION_CONFIDENCE = 0.30;

// ECG quality multipliers.
//
// GOOD:
//   Full confidence is allowed.
//
// USABLE:
//   Signal is usable but less trustworthy than GOOD.
//
// NOISY:
//   Strong down-weighting prevents noisy ECG from dominating fusion.
//
// MOTION_CORRUPTED / INVALID:
//   Do not allow the ECG candidate to contribute meaningful confidence.
const SQI_CONFIDENCE_MULTIPLIER: Record<
  SQI["label"],
  number
> = {
  GOOD: 1.00,
  USABLE: 0.85,
  NOISY: 0.35,
  MOTION_CORRUPTED: 0.10,
  INVALID: 0.00,
};

// During active motion, cardiac measurements can be physiologically valid,
// but they are more difficult to interpret for anomaly/risk purposes.
//
// We therefore reduce confidence rather than blindly rejecting the data.
//
// This is consistent with the architecture's motion gate: motion should
// reduce false positives rather than erase all physiological information.
const CARDIAC_MOTION_MULTIPLIER: Record<
  ActivityState,
  number
> = {
  REST: 1.00,
  LIGHT: 0.85,
  ACTIVE: 0.55,
};

// Environmental risk can legitimately increase during activity, so we do NOT
// arbitrarily suppress heat/respiratory risk during ACTIVE.
//
// Their upstream modules are responsible for incorporating activity into
// their own risk scores. Fusion only uses motion as contextual evidence.
const VALID_RISK_LEVELS: RiskLevel[] = [
  "NORMAL",
  "CAUTION",
  "RISK",
];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function isFiniteNumber(
  value: number | null | undefined,
): value is number {
  return (
    value !== null &&
    value !== undefined &&
    Number.isFinite(value)
  );
}

/**
 * Keeps confidence inside the documented [0, 1] range.
 *
 * Why:
 * Confidence is metadata, not a sensor measurement. A malformed upstream
 * value must never propagate into a confidence >1 or <0.
 */
function clampConfidence(
  confidence: number,
): number {
  if (!Number.isFinite(confidence)) {
    return 0;
  }

  return Math.max(
    0,
    Math.min(1, confidence),
  );
}

/**
 * Returns whether a numeric candidate can legitimately participate in fusion.
 *
 * Why:
 * null means "unavailable" throughout this engine. We must never turn null
 * into zero because zero could be interpreted as a genuine low-risk score.
 */
function hasUsableValue(
  value: number | null,
): value is number {
  return (
    value !== null &&
    Number.isFinite(value)
  );
}

/**
 * Adds a bounded evidence item.
 *
 * Evidence is intentionally human-readable because the downstream decision
 * engine needs to explain why a risk candidate was accepted or weakened.
 */
function addEvidence(
  evidence: string[],
  message: string,
): void {
  if (!evidence.includes(message)) {
    evidence.push(message);
  }
}

// ---------------------------------------------------------------------------
// Category fusion
// ---------------------------------------------------------------------------

/**
 * Creates the fused representation for a category that has one upstream
 * risk candidate.
 *
 * Because the current FusionInput contains only one candidate, the "fusion"
 * operation is effectively confidence qualification + contextual weighting.
 *
 * This is preferable to fabricating a second sensor contribution.
 */
function fuseSingleCandidate(
  category: "cardiac" | "heat" | "respiratory",
  value: number | null,
  upstreamConfidence: number,
  evidence: string[],
  confidenceMultiplier = 1,
): FusedCategoryRisk {
  const confidence =
    clampConfidence(
      upstreamConfidence,
    );

  if (!hasUsableValue(value)) {
    return {
      score: null,
      confidence: 0,
      evidence: [
        ...evidence,
        `${category}: candidate unavailable`,
      ],
      insufficientData: true,
    };
  }

  if (confidence <= 0) {
    return {
      score: null,
      confidence: 0,
      evidence: [
        ...evidence,
        `${category}: candidate confidence is zero`,
      ],
      insufficientData: true,
    };
  }

  const adjustedConfidence =
    clampConfidence(
      confidence * confidenceMultiplier,
    );

  if (
    adjustedConfidence <
    MIN_FUSION_CONFIDENCE
  ) {
    return {
      score: null,
      confidence: adjustedConfidence,
      evidence: [
        ...evidence,
        `${category}: insufficient confidence after fusion`,
      ],
      insufficientData: true,
    };
  }

  return {
    score: value,
    confidence: adjustedConfidence,
    evidence,
    insufficientData: false,
  };
}

// ---------------------------------------------------------------------------
// Cardiac fusion
// ---------------------------------------------------------------------------

function fuseCardiac(
  input: FusionInput,
): FusedCategoryRisk {
  const evidence: string[] = [];

  const sqi =
    input.cardiac.sqi;

  const sqiMultiplier =
    SQI_CONFIDENCE_MULTIPLIER[
      sqi.label
    ];

  const motionMultiplier =
    CARDIAC_MOTION_MULTIPLIER[
      input.motion.activityState
    ];

  if (sqi.label === "GOOD") {
    addEvidence(
      evidence,
      "ECG signal quality is GOOD",
    );
  } else if (sqi.label === "USABLE") {
    addEvidence(
      evidence,
      "ECG signal quality is USABLE",
    );
  } else {
    addEvidence(
      evidence,
      `ECG signal quality is ${sqi.label}`,
    );
  }

  if (
    input.motion.activityState ===
    "ACTIVE"
  ) {
    addEvidence(
      evidence,
      "Active motion reduces cardiac confidence",
    );
  } else if (
    input.motion.activityState ===
    "LIGHT"
  ) {
    addEvidence(
      evidence,
      "Light motion slightly reduces cardiac confidence",
    );
  } else {
    addEvidence(
      evidence,
      "Resting motion state supports cardiac interpretation",
    );
  }

  const adjustedMultiplier =
    sqiMultiplier *
    motionMultiplier;

  return fuseSingleCandidate(
    "cardiac",
    input.cardiac.value,
    input.cardiac.confidence,
    evidence,
    adjustedMultiplier,
  );
}

// ---------------------------------------------------------------------------
// Heat fusion
// ---------------------------------------------------------------------------

function fuseHeat(
  input: FusionInput,
): FusedCategoryRisk {
  const evidence: string[] = [];

  if (
    input.motion.activityState ===
    "ACTIVE"
  ) {
    addEvidence(
      evidence,
      "Active motion is relevant context for heat exposure",
    );
  } else if (
    input.motion.activityState ===
    "LIGHT"
  ) {
    addEvidence(
      evidence,
      "Light activity is present",
    );
  } else {
    addEvidence(
      evidence,
      "Resting state detected",
    );
  }

  return fuseSingleCandidate(
    "heat",
    input.heat.value,
    input.heat.confidence,
    evidence,
  );
}

// ---------------------------------------------------------------------------
// Respiratory fusion
// ---------------------------------------------------------------------------

function fuseRespiratory(
  input: FusionInput,
): FusedCategoryRisk {
  const evidence: string[] = [];

  if (
    input.motion.activityState ===
    "ACTIVE"
  ) {
    addEvidence(
      evidence,
      "Active motion is relevant context for respiratory exposure",
    );
  } else if (
    input.motion.activityState ===
    "LIGHT"
  ) {
    addEvidence(
      evidence,
      "Light activity is present",
    );
  } else {
    addEvidence(
      evidence,
      "Resting state detected",
    );
  }

  return fuseSingleCandidate(
    "respiratory",
    input.respiratory.value,
    input.respiratory.confidence,
    evidence,
  );
}

// ---------------------------------------------------------------------------
// Fall fusion
// ---------------------------------------------------------------------------

function fuseFall(
  input: FusionInput,
): FusionOutput["fall"] {
  const fall =
    input.motion.fall;

  // Fall detection is already a state-machine result from the motion
  // pipeline. We do not re-detect or override it here.
  //
  // Why:
  // Re-running fall logic in fusion would duplicate the source of truth and
  // could create disagreement between the motion and fusion layers.
  return {
    detected: Boolean(
      fall.detected,
    ),
    confidence:
      clampConfidence(
        fall.confidence,
      ),
    stage: fall.stage,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Fuse the already-derived risk candidates.
 *
 * The function is deliberately pure:
 *   same input -> same output
 *
 * Why:
 * A pure fusion function is easy to unit-test with deterministic synthetic
 * sensor scenarios and is safe to run inside the React Native JS runtime.
 */
export function fuseSensorRisks(
  input: FusionInput,
): FusionOutput {
  if (!input) {
    // This branch is defensive. The TypeScript contract says input is
    // required, but runtime JavaScript can still pass undefined/null.
    //
    // We cannot manufacture meaningful category scores without data.
    return {
      cardiac: {
        score: null,
        confidence: 0,
        evidence: [
          "Fusion input unavailable",
        ],
        insufficientData: true,
      },
      heat: {
        score: null,
        confidence: 0,
        evidence: [
          "Fusion input unavailable",
        ],
        insufficientData: true,
      },
      respiratory: {
        score: null,
        confidence: 0,
        evidence: [
          "Fusion input unavailable",
        ],
        insufficientData: true,
      },
      fall: {
        detected: false,
        confidence: 0,
        stage: "NONE",
      },
    };
  }

  return {
    cardiac: fuseCardiac(input),
    heat: fuseHeat(input),
    respiratory:
      fuseRespiratory(input),
    fall: fuseFall(input),
  };
}

// ---------------------------------------------------------------------------
// Optional compatibility alias
// ---------------------------------------------------------------------------
//
// Keep one obvious public name while allowing the implementation name to
// remain descriptive inside tests or future orchestration code.
//
// Both functions have exactly the same contract.
export const sensorFusion =
  fuseSensorRisks;

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

/**
 * Runtime validation for a risk candidate.
 *
 * This is intentionally not exported as a new public type. It only protects
 * this module from malformed runtime values.
 */
function isValidRiskCandidate(
  value: number | null,
): boolean {
  return (
    value === null ||
    (
      Number.isFinite(value) &&
      value >= 0 &&
      value <= 1
    )
  );
}

/**
 * Validates the numeric risk/confidence portions of FusionInput.
 *
 * Useful before processing data arriving from JavaScript/JSON boundaries.
 *
 * It does not throw because the engine's safety rule is graceful degradation:
 * invalid data should become "insufficient data", not crash the risk pipeline.
 */
export function validateFusionInput(
  input: FusionInput,
): boolean {
  if (!input) {
    return false;
  }

  if (
    !isValidRiskCandidate(
      input.cardiac.value,
    ) ||
    !isValidRiskCandidate(
      input.heat.value,
    ) ||
    !isValidRiskCandidate(
      input.respiratory.value,
    )
  ) {
    return false;
  }

  if (
    !Number.isFinite(
      input.cardiac.confidence,
    ) ||
    !Number.isFinite(
      input.heat.confidence,
    ) ||
    !Number.isFinite(
      input.respiratory.confidence,
    )
  ) {
    return false;
  }

  if (
    input.cardiac.confidence < 0 ||
    input.cardiac.confidence > 1 ||
    input.heat.confidence < 0 ||
    input.heat.confidence > 1 ||
    input.respiratory.confidence < 0 ||
    input.respiratory.confidence > 1
  ) {
    return false;
  }

  if (
    !VALID_RISK_LEVELS ||
    input.motion.activityState ===
      undefined
  ) {
    return false;
  }

  return true;
}