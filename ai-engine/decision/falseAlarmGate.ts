// ============================================================================
// ai-engine/decision/falseAlarmGate.ts
//
// Phase 1 False-Alarm Gate.
//
// This is the safety barrier between:
//     "the algorithms see something unusual"
// and:
//     "the system is allowed to raise an alert"
//
// Architecture:
//   1. Signal quality
//   2. Motion gate
//   3. Temporal persistence
//   4. Personal baseline comparison
//   5. Cross-sensor support
//   6. Confidence threshold
//   7. Hysteresis + cooldown
//
// IMPORTANT:
// The public input/output contracts come directly from types.ts.
// Persistence, previous-alert state, and cooldown state are deliberately
// maintained internally because the current FalseAlarmGateInput contract
// does not expose those fields.
//
// Phase 1 uses deterministic rule-based logic only.
// No ML/DL, no React Native, no Node-native dependencies.
// ============================================================================

import type {
  ActivityState,
  FalseAlarmGateInput,
  FalseAlarmGateOutput,
  SQI,
} from "../types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
//
// These are engineering thresholds for the Phase-1 prototype.
//
// They are NOT clinical diagnostic thresholds.
//
// The purpose is to prevent unstable, one-window candidates from becoming
// alerts. They should be recalibrated after real-world validation data exists.
// ---------------------------------------------------------------------------

export interface FalseAlarmGateOptions {
  /**
   * Minimum score at which a candidate is considered alert-worthy.
   *
   * Scores below this are not sent through the alert path.
   */
  alertScoreThreshold?: number;

  /**
   * Number of consecutive qualifying observations required.
   *
   * Two consecutive windows follows the Phase-1 architecture.
   */
  persistenceWindows?: number;

  /**
   * Minimum confidence required after all confidence adjustments.
   */
  minimumConfidence?: number;

  /**
   * Minimum absolute baseline deviation required when a baseline value
   * is supplied.
   *
   * The value is expressed in the same normalized/deviation scale supplied
   * by the caller.
   */
  minimumBaselineDeviation?: number;

  /**
   * Score at which an already-active alert is allowed to clear.
   *
   * Lower than alertScoreThreshold implements hysteresis.
   */
  clearScoreThreshold?: number;

  /**
   * Minimum milliseconds between repeated alerts for one category.
   */
  cooldownMs?: number;
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

interface CategoryState {
  /**
   * Number of consecutive qualifying candidate windows.
   */
  persistenceCount: number;

  /**
   * Whether the category currently has an active alert.
   */
  alertActive: boolean;

  /**
   * Timestamp of the last emitted alert.
   */
  lastAlertTimestamp: number | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_ALERT_SCORE_THRESHOLD = 0.70;

const DEFAULT_PERSISTENCE_WINDOWS = 2;

const DEFAULT_MINIMUM_CONFIDENCE = 0.70;

/**
 * A baseline deviation is only meaningful when it reaches this magnitude.
 *
 * This is deliberately conservative and should not be interpreted as a
 * medical threshold. It simply prevents tiny deviations from satisfying
 * the baseline stage of the safety gate.
 */
const DEFAULT_MINIMUM_BASELINE_DEVIATION = 2.0;

/**
 * Hysteresis:
 *
 * trigger >= 0.70
 * clear   <= 0.50
 *
 * This prevents:
 *
 *   0.71 -> alert
 *   0.69 -> clear
 *   0.71 -> alert
 *
 * from producing notification flicker.
 */
const DEFAULT_CLEAR_SCORE_THRESHOLD = 0.50;

const DEFAULT_COOLDOWN_MS = 5 * 60 * 1000;

// ---------------------------------------------------------------------------
// SQI helpers
// ---------------------------------------------------------------------------

/**
 * Returns whether the ECG quality is sufficient for the gate.
 *
 * Why:
 * The architecture explicitly says anything below USABLE must not be
 * allowed to produce a strong alert. :contentReference[oaicite:2]{index=2}
 */
function isSQIUsable(
  sqi: SQI | undefined,
): boolean {
  if (!sqi) {
    return true;
  }

  return (
    sqi.label === "GOOD" ||
    sqi.label === "USABLE"
  );
}

/**
 * Returns a readable SQI evidence string.
 */
function sqiEvidence(
  sqi: SQI | undefined,
): string | null {
  if (!sqi) {
    return null;
  }

  return `ECG SQI: ${sqi.label} (${sqi.score.toFixed(2)})`;
}

// ---------------------------------------------------------------------------
// Category state
// ---------------------------------------------------------------------------

type GateCategory =
  | "cardiac"
  | "heat"
  | "respiratory"
  | "fall";

function createCategoryState(): CategoryState {
  return {
    persistenceCount: 0,
    alertActive: false,
    lastAlertTimestamp: null,
  };
}

// ---------------------------------------------------------------------------
// False Alarm Gate
// ---------------------------------------------------------------------------

export class FalseAlarmGate {
  private readonly alertScoreThreshold: number;

  private readonly persistenceWindows: number;

  private readonly minimumConfidence: number;

  private readonly minimumBaselineDeviation: number;

  private readonly clearScoreThreshold: number;

  private readonly cooldownMs: number;

  private readonly states: Record<
    GateCategory,
    CategoryState
  >;

  constructor(
    options: FalseAlarmGateOptions = {},
  ) {
    this.alertScoreThreshold =
      clamp01(
        options.alertScoreThreshold ??
          DEFAULT_ALERT_SCORE_THRESHOLD,
      );

    this.persistenceWindows =
      Math.max(
        1,
        Math.floor(
          options.persistenceWindows ??
            DEFAULT_PERSISTENCE_WINDOWS,
        ),
      );

    this.minimumConfidence =
      clamp01(
        options.minimumConfidence ??
          DEFAULT_MINIMUM_CONFIDENCE,
      );

    this.minimumBaselineDeviation =
      Math.max(
        0,
        options.minimumBaselineDeviation ??
          DEFAULT_MINIMUM_BASELINE_DEVIATION,
      );

    this.clearScoreThreshold =
      clamp01(
        options.clearScoreThreshold ??
          DEFAULT_CLEAR_SCORE_THRESHOLD,
      );

    this.cooldownMs =
      Math.max(
        0,
        options.cooldownMs ??
          DEFAULT_COOLDOWN_MS,
      );

    this.states = {
      cardiac: createCategoryState(),
      heat: createCategoryState(),
      respiratory: createCategoryState(),
      fall: createCategoryState(),
    };
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Evaluate one candidate window.
   *
   * IMPORTANT:
   * The caller should invoke this once per analysis window, not once per
   * individual ECG sample or sensor packet.
   *
   * Why:
   * Persistence is meaningful across physiological analysis windows, not
   * across individual samples.
   */
  evaluate(
    input: FalseAlarmGateInput,
    timestamp: number,
  ): FalseAlarmGateOutput {
    const evidence: string[] = [];

    const state =
      this.states[input.category];

    // -----------------------------------------------------------------------
    // Basic runtime validation
    // -----------------------------------------------------------------------

    if (
      !Number.isFinite(
        input.candidateScore,
      ) ||
      !Number.isFinite(input.confidence) ||
      !Number.isFinite(timestamp)
    ) {
      this.resetPersistence(
        input.category,
      );

      return {
        shouldAlert: false,
        suppressedReason:
          "LOW_CONFIDENCE",
        evidence: [
          "Invalid candidate or confidence data",
        ],
      };
    }

    const score = clamp01(
      input.candidateScore,
    );

    const confidence = clamp01(
      input.confidence,
    );

    // -----------------------------------------------------------------------
    // 1. SIGNAL QUALITY GATE
    // -----------------------------------------------------------------------

    if (
      input.category === "cardiac" &&
      !isSQIUsable(input.sqi)
    ) {
      this.resetPersistence(
        input.category,
      );

      const sqiText =
        sqiEvidence(input.sqi);

      return {
        shouldAlert: false,
        suppressedReason:
          "LOW_QUALITY",
        evidence: [
          ...(sqiText
            ? [sqiText]
            : []),
          "Cardiac candidate suppressed because ECG quality is below USABLE",
        ],
      };
    }

    const sqiText =
      sqiEvidence(input.sqi);

    if (sqiText) {
      evidence.push(sqiText);
    }

    // -----------------------------------------------------------------------
    // 2. MOTION GATE
    // -----------------------------------------------------------------------
    //
    // Cardiac anomalies are much harder to interpret during vigorous motion.
    //
    // We therefore block alert persistence during ACTIVE motion.
    //
    // Heat and respiratory risks are NOT blocked here because activity itself
    // can legitimately increase those risks.
    // -----------------------------------------------------------------------

    if (
      input.category === "cardiac" &&
      input.activityState === "ACTIVE"
    ) {
      this.resetPersistence(
        input.category,
      );

      return {
        shouldAlert: false,
        suppressedReason: "MOTION",
        evidence: [
          ...evidence,
          "Cardiac candidate suppressed during ACTIVE motion",
        ],
      };
    }

    if (
      input.activityState === "REST"
    ) {
      evidence.push(
        "Motion state: REST",
      );
    } else if (
      input.activityState === "LIGHT"
    ) {
      evidence.push(
        "Motion state: LIGHT",
      );
    } else if (
      input.activityState === "ACTIVE"
    ) {
      evidence.push(
        "Motion state: ACTIVE",
      );
    }

    // -----------------------------------------------------------------------
    // 3. CANDIDATE THRESHOLD
    // -----------------------------------------------------------------------
    //
    // Candidates below the alert threshold are not allowed to accumulate
    // persistence toward an alert.
    //
    // Why:
    // Otherwise several weak windows could eventually create a strong alert
    // even though no individual window actually crossed the alert boundary.
    // -----------------------------------------------------------------------

    if (
      score <
      this.alertScoreThreshold
    ) {
      this.handlePotentialClear(
        state,
        score,
      );

      return {
        shouldAlert: false,
        suppressedReason:
          "INSUFFICIENT_PERSISTENCE",
        evidence: [
          ...evidence,
          `Candidate score ${score.toFixed(2)} is below alert threshold ${this.alertScoreThreshold.toFixed(2)}`,
        ],
      };
    }

    evidence.push(
      `Candidate score ${score.toFixed(2)} crossed alert threshold`,
    );

    // -----------------------------------------------------------------------
    // 4. PERSONAL BASELINE GATE
    // -----------------------------------------------------------------------
    //
    // This is intentionally strict.
    //
    // The current types.ts contract gives us:
    //
    //     baselineDeviation?: number
    //
    // but does NOT give us:
    //
    //     baselineReady: boolean
    //
    // Therefore:
    //
    //     undefined = baseline stage has NOT passed
    //
    // We must not interpret missing baseline data as zero deviation.
    // -----------------------------------------------------------------------

    if (
      input.baselineDeviation ===
      undefined ||
      !Number.isFinite(
        input.baselineDeviation,
      )
    ) {
      this.resetPersistence(
        input.category,
      );

      return {
        shouldAlert: false,
        suppressedReason:
          "LOW_CONFIDENCE",
        evidence: [
          ...evidence,
          "Personal baseline comparison unavailable",
          "Alert blocked until a valid baseline deviation is available",
        ],
      };
    }

    const baselineDeviation =
      Math.abs(
        input.baselineDeviation,
      );

    evidence.push(
      `Baseline deviation: ${input.baselineDeviation.toFixed(2)}`,
    );

    if (
      baselineDeviation <
      this.minimumBaselineDeviation
    ) {
      this.resetPersistence(
        input.category,
      );

      return {
        shouldAlert: false,
        suppressedReason:
          "LOW_CONFIDENCE",
        evidence: [
          ...evidence,
          `Baseline deviation ${baselineDeviation.toFixed(2)} is below required ${this.minimumBaselineDeviation.toFixed(2)}`,
        ],
      };
    }

    evidence.push(
      "Personal baseline deviation passed",
    );

    // -----------------------------------------------------------------------
    // 5. TEMPORAL PERSISTENCE
    // -----------------------------------------------------------------------
    //
    // Persistence is tracked per category.
    //
    // A candidate must survive multiple consecutive analysis windows.
    // One abnormal window is never enough.
    // -----------------------------------------------------------------------

    state.persistenceCount += 1;

    evidence.push(
      `Persistence: ${state.persistenceCount}/${this.persistenceWindows} qualifying windows`,
    );

    if (
      state.persistenceCount <
      this.persistenceWindows
    ) {
      return {
        shouldAlert: false,
        suppressedReason:
          "INSUFFICIENT_PERSISTENCE",
        evidence,
      };
    }

    // -----------------------------------------------------------------------
    // 6. CROSS-SENSOR SUPPORT
    // -----------------------------------------------------------------------
    //
    // Cross-sensor support is evidence, not an absolute requirement.
    //
    // Why:
    // Cardiac risk may legitimately be driven by ECG alone. Conversely,
    // heat/respiratory candidates can become more convincing when physiological
    // and environmental evidence agree.
    // -----------------------------------------------------------------------

    if (
      input.crossSensorSupport === true
    ) {
      evidence.push(
        "Independent sensor/context support present",
      );
    } else {
      evidence.push(
        "No independent cross-sensor support",
      );
    }

    // -----------------------------------------------------------------------
    // 7. CONFIDENCE GATE
    // -----------------------------------------------------------------------

    if (
      confidence <
      this.minimumConfidence
    ) {
      return {
        shouldAlert: false,
        suppressedReason:
          "LOW_CONFIDENCE",
        evidence: [
          ...evidence,
          `Confidence ${confidence.toFixed(2)} is below required ${this.minimumConfidence.toFixed(2)}`,
        ],
      };
    }

    evidence.push(
      `Confidence ${confidence.toFixed(2)} passed threshold`,
    );

    // -----------------------------------------------------------------------
    // 8. COOLDOWN
    // -----------------------------------------------------------------------

    if (
      state.lastAlertTimestamp !==
      null
    ) {
      const elapsed =
        timestamp -
        state.lastAlertTimestamp;

      if (
        elapsed >= 0 &&
        elapsed < this.cooldownMs
      ) {
        return {
          shouldAlert: false,
          suppressedReason: "COOLDOWN",
          evidence: [
            ...evidence,
            `Alert cooldown active (${Math.ceil((this.cooldownMs - elapsed) / 1000)}s remaining)`,
          ],
        };
      }
    }

    // -----------------------------------------------------------------------
    // 9. HYSTERESIS
    // -----------------------------------------------------------------------
    //
    // If an alert is already active, it should not repeatedly fire just
    // because the score crosses the same boundary.
    //
    // The active state remains until the score drops below the lower
    // clear threshold.
    // -----------------------------------------------------------------------

    if (
      state.alertActive
    ) {
      if (
        score >
        this.clearScoreThreshold
      ) {
        return {
          shouldAlert: false,
          suppressedReason: null,
          evidence: [
            ...evidence,
            "Alert remains active under hysteresis",
          ],
        };
      }

      state.alertActive = false;
      state.persistenceCount = 0;

      return {
        shouldAlert: false,
        suppressedReason: null,
        evidence: [
          ...evidence,
          `Score ${score.toFixed(2)} reached hysteresis clear region`,
        ],
      };
    }

    // -----------------------------------------------------------------------
    // ALERT ALLOWED
    // -----------------------------------------------------------------------

    state.alertActive = true;
    state.lastAlertTimestamp =
      timestamp;

    state.persistenceCount = 0;

    return {
      shouldAlert: true,
      suppressedReason: null,
      evidence: [
        ...evidence,
        "All false-alarm gates passed",
        "Alert permitted",
      ],
    };
  }

  // -------------------------------------------------------------------------
  // State management
  // -------------------------------------------------------------------------

  /**
   * Clears only persistence.
   *
   * Why:
   * A temporary bad-quality/motion window should not necessarily erase the
   * historical cooldown timestamp.
   */
  private resetPersistence(
    category: GateCategory,
  ): void {
    this.states[
      category
    ].persistenceCount = 0;
  }

  private handlePotentialClear(
    state: CategoryState,
    score: number,
  ): void {
    if (
      state.alertActive &&
      score <=
        this.clearScoreThreshold
    ) {
      state.alertActive = false;
      state.persistenceCount = 0;
    } else {
      state.persistenceCount = 0;
    }
  }

  /**
   * Resets all internal state.
   *
   * Useful when:
   * - a new user starts
   * - a monitoring session ends
   * - the device is reinitialized
   */
  reset(): void {
    for (
      const category of [
        "cardiac",
        "heat",
        "respiratory",
        "fall",
      ] as GateCategory[]
    ) {
      this.states[
        category
      ] = createCategoryState();
    }
  }

  /**
   * Reset one category without affecting the others.
   */
  resetCategory(
    category: GateCategory,
  ): void {
    this.states[
      category
    ] = createCategoryState();
  }
}

// ---------------------------------------------------------------------------
// Functional singleton-style helper
// ---------------------------------------------------------------------------
//
// The class is the preferred API when a long-running monitoring session is
// active because persistence/cooldown state must survive multiple calls.
//
// This factory keeps creation explicit and avoids a hidden global singleton,
// which would be dangerous if multiple users/sessions were ever processed in
// the same JS runtime.
// ---------------------------------------------------------------------------

export function createFalseAlarmGate(
  options: FalseAlarmGateOptions = {},
): FalseAlarmGate {
  return new FalseAlarmGate(
    options,
  );
}

// ---------------------------------------------------------------------------
// Numeric helper
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