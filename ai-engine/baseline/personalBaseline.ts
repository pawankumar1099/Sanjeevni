// ============================================================================
// ai-engine/baseline/personalBaseline.ts
//
// Personal baseline engine for Sanjeevni.
//
// Purpose:
// - Learn the user's normal physiological/environmental range over time.
// - Maintain robust rolling statistics.
// - Maintain an EWMA (Exponentially Weighted Moving Average) so the baseline
//   can slowly adapt to genuine long-term changes.
// - Prevent baseline-based anomaly decisions during cold start.
//
// IMPORTANT:
// - This module does NOT make a medical diagnosis.
// - This module does NOT decide whether a user is at risk.
// - It only describes what is normal for this user.
// - Risk/alert decisions belong to fusion/ and decision/.
//
// Phase 1 constraint:
// DSP + statistics + rule-based logic only.
// No ML/DL models.
//
// Architecture rule:
// The implementation is stateful because a personal baseline must accumulate
// information across sensor ticks. The public methods expose statistics rather
// than implementation details, so the internals can later be replaced by an
// ML-based personalization model without changing downstream concepts.
// ============================================================================

import { clamp, mean, stdDev } from "../utils/math";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * Metrics that can maintain an independent personal baseline.
 *
 * HR and RR are the most important cardiac personalization metrics.
 * HRV metrics are kept separate because their physiological scale and
 * interpretation differ from heart rate.
 *
 * Temperature is included because Sanjeevni has environmental sensing.
 */
export type BaselineMetric =
  | "heartRate"
  | "rrInterval"
  | "rmssd"
  | "sdnn"
  | "temperature";

/**
 * Current statistics for one personal metric.
 */
export interface BaselineMetricState {
  /**
   * Number of accepted observations used by this baseline.
   */
  sampleCount: number;

  /**
   * Arithmetic mean of the retained observations.
   *
   * null means that no valid baseline exists yet.
   */
  mean: number | null;

  /**
   * Population standard deviation of the retained observations.
   *
   * null means that there is insufficient data.
   */
  stdDev: number | null;

  /**
   * Exponentially weighted moving average.
   *
   * null during cold start.
   */
  ewma: number | null;

  /**
   * Most recent accepted observation.
   *
   * null when no observation has been accepted.
   */
  lastValue: number | null;

  /**
   * Whether this metric has enough clean observations to be
   * considered a usable personal baseline.
   */
  ready: boolean;

  /**
   * Confidence in the baseline itself.
   *
   * This is NOT medical confidence.
   * It describes how much clean historical data has accumulated.
   */
  confidence: number;
}

/**
 * Complete personal baseline snapshot.
 */
export interface PersonalBaselineSnapshot {
  heartRate: BaselineMetricState;
  rrInterval: BaselineMetricState;
  rmssd: BaselineMetricState;
  sdnn: BaselineMetricState;
  temperature: BaselineMetricState;
}

/**
 * Result of comparing a new observation against a personal baseline.
 */
export interface BaselineComparison {
  /**
   * Difference between observation and baseline mean.
   *
   * null when baseline is not ready.
   */
  deviation: number | null;

  /**
   * Absolute standardized deviation.
   *
   * null when baseline variance is unavailable.
   */
  zScore: number | null;

  /**
   * Whether the baseline is sufficiently established to compare.
   */
  baselineReady: boolean;

  /**
   * Confidence in the comparison.
   *
   * 0 when the baseline is unavailable.
   */
  confidence: number;
}

/**
 * Configuration for the personal baseline engine.
 */
export interface PersonalBaselineOptions {
  /**
   * Number of observations required before a baseline becomes
   * usable for anomaly comparison.
   *
   * Default = 30.
   *
   * WHY:
   * A handful of readings can be dominated by transient activity,
   * sensor noise, or poor placement. We deliberately require a
   * meaningful history before downstream anomaly logic can call
   * something "unusual for this person".
   */
  minSamplesForBaseline?: number;

  /**
   * Maximum number of observations retained per metric.
   *
   * Default = 300.
   *
   * WHY:
   * A bounded rolling window prevents unbounded memory growth on
   * a phone while still providing enough history to estimate
   * personal variability.
   */
  maxSamplesPerMetric?: number;

  /**
   * EWMA smoothing factor.
   *
   * Must be > 0 and <= 1.
   *
   * Smaller values adapt more slowly.
   *
   * Default = 0.05.
   */
  ewmaAlpha?: number;

  /**
   * Maximum allowed absolute z-score before an observation is
   * considered an extreme outlier for baseline learning.
   *
   * Default = 4.
   *
   * WHY:
   * A very abnormal/noisy observation should not immediately move
   * the user's baseline toward itself.
   */
  outlierZScore?: number;

  /**
   * Minimum standard deviation used for z-score calculation.
   *
   * This prevents division by a tiny variance when a metric is
   * temporarily almost constant.
   *
   * Default = 0.001.
   */
  minimumStdDev?: number;
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

interface MetricBuffer {
  values: number[];
  ewma: number | null;
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Validates a numeric observation.
 *
 * WHY:
 * Missing/invalid data must never enter the baseline as zero.
 * Zero would become indistinguishable from a legitimate physiological
 * value and would corrupt the user's learned baseline.
 */
function isValidObservation(
  value: number | null | undefined,
): value is number {
  return (
    value !== null &&
    value !== undefined &&
    Number.isFinite(value)
  );
}

/**
 * Creates an empty metric state.
 */
function createMetricBuffer(): MetricBuffer {
  return {
    values: [],
    ewma: null,
  };
}

/**
 * Converts an internal metric buffer into its public state.
 */
function toMetricState(
  buffer: MetricBuffer,
  minSamples: number,
): BaselineMetricState {
  const sampleCount = buffer.values.length;

  if (sampleCount === 0) {
    return {
      sampleCount: 0,
      mean: null,
      stdDev: null,
      ewma: null,
      lastValue: null,
      ready: false,
      confidence: 0,
    };
  }

  const currentMean = mean(buffer.values);
  const currentStdDev = stdDev(buffer.values);

  /**
   * Confidence grows with sample count and reaches 1 only after
   * the configured minimum baseline history exists.
   *
   * This is a data-sufficiency confidence, not a clinical confidence.
   */
  const confidence = clamp(
    sampleCount / minSamples,
    0,
    1,
  );

  return {
    sampleCount,
    mean: currentMean,
    stdDev: currentStdDev,
    ewma: buffer.ewma,
    lastValue:
      buffer.values[buffer.values.length - 1] ?? null,
    ready:
      sampleCount >= minSamples &&
      currentMean !== null &&
      currentStdDev !== null,
    confidence,
  };
}

// ---------------------------------------------------------------------------
// PersonalBaselineEngine
// ---------------------------------------------------------------------------

/**
 * Stateful personal baseline engine.
 */
export class PersonalBaselineEngine {
  private readonly minSamplesForBaseline: number;

  private readonly maxSamplesPerMetric: number;

  private readonly ewmaAlpha: number;

  private readonly outlierZScore: number;

  private readonly minimumStdDev: number;

  private readonly metrics: Record<
    BaselineMetric,
    MetricBuffer
  >;

  constructor(
    options: PersonalBaselineOptions = {},
  ) {
    this.minSamplesForBaseline =
      Math.max(
        1,
        Math.floor(
          options.minSamplesForBaseline ?? 30,
        ),
      );

    this.maxSamplesPerMetric =
      Math.max(
        this.minSamplesForBaseline,
        Math.floor(
          options.maxSamplesPerMetric ?? 300,
        ),
      );

    this.ewmaAlpha = clamp(
      options.ewmaAlpha ?? 0.05,
      0.001,
      1,
    );

    this.outlierZScore = Math.max(
      1,
      options.outlierZScore ?? 4,
    );

    this.minimumStdDev = Math.max(
      Number.EPSILON,
      options.minimumStdDev ?? 0.001,
    );

    this.metrics = {
      heartRate: createMetricBuffer(),
      rrInterval: createMetricBuffer(),
      rmssd: createMetricBuffer(),
      sdnn: createMetricBuffer(),
      temperature: createMetricBuffer(),
    };
  }

  // -------------------------------------------------------------------------
  // Data ingestion
  // -------------------------------------------------------------------------

  /**
   * Add one observation to a personal metric.
   *
   * Returns true when the observation was accepted.
   *
   * Returns false when the observation was rejected because it is invalid
   * or appears to be an extreme outlier.
   */
  addObservation(
    metric: BaselineMetric,
    value: number | null | undefined,
  ): boolean {
    if (!isValidObservation(value)) {
      return false;
    }

    const buffer = this.metrics[metric];

    /**
     * During cold start we do not yet have enough information to
     * calculate a reliable personal variance.
     *
     * Therefore the first observations are accepted directly.
     */
    if (buffer.values.length < this.minSamplesForBaseline) {
      this.appendValue(buffer, value);
      return true;
    }

    /**
     * Once a baseline exists, protect it against extreme outliers.
     *
     * WHY:
     * If a noisy ECG window suddenly produces an impossible HR,
     * feeding that value directly into the baseline would shift the
     * baseline toward the artifact and make future detection worse.
     */
    const baselineMean = mean(buffer.values);
    const baselineStdDev = stdDev(buffer.values);

    if (
      baselineMean !== null &&
      baselineStdDev !== null &&
      baselineStdDev > this.minimumStdDev
    ) {
      const z =
        Math.abs(
          (value - baselineMean) /
            baselineStdDev,
        );

      if (
        Number.isFinite(z) &&
        z > this.outlierZScore
      ) {
        return false;
      }
    }

    this.appendValue(buffer, value);

    return true;
  }

  /**
   * Add multiple observations to one metric.
   *
   * Useful when a clean ECG window produces multiple RR intervals.
   */
  addObservations(
    metric: BaselineMetric,
    values: Array<number | null | undefined>,
  ): number {
    let accepted = 0;

    for (const value of values) {
      if (this.addObservation(metric, value)) {
        accepted += 1;
      }
    }

    return accepted;
  }

  /**
   * Internal bounded-buffer update.
   */
  private appendValue(
    buffer: MetricBuffer,
    value: number,
  ): void {
    /**
     * Update EWMA incrementally.
     *
     * EWMA is useful here because it adapts gradually to genuine
     * long-term changes without reacting as aggressively as a
     * simple latest-value baseline.
     */
    if (buffer.ewma === null) {
      buffer.ewma = value;
    } else {
      buffer.ewma =
        this.ewmaAlpha * value +
        (1 - this.ewmaAlpha) * buffer.ewma;
    }

    buffer.values.push(value);

    /**
     * Keep memory bounded.
     *
     * The oldest observations are removed first, producing a
     * rolling personal baseline rather than an ever-growing lifetime
     * average.
     */
    if (
      buffer.values.length >
      this.maxSamplesPerMetric
    ) {
      buffer.values.shift();
    }
  }

  // -------------------------------------------------------------------------
  // Baseline access
  // -------------------------------------------------------------------------

  /**
   * Return the current state for one metric.
   */
  getMetricState(
    metric: BaselineMetric,
  ): BaselineMetricState {
    return toMetricState(
      this.metrics[metric],
      this.minSamplesForBaseline,
    );
  }

  /**
   * Return the complete personal baseline.
   */
  getSnapshot(): PersonalBaselineSnapshot {
    return {
      heartRate: this.getMetricState(
        "heartRate",
      ),
      rrInterval: this.getMetricState(
        "rrInterval",
      ),
      rmssd: this.getMetricState("rmssd"),
      sdnn: this.getMetricState("sdnn"),
      temperature: this.getMetricState(
        "temperature",
      ),
    };
  }

  // -------------------------------------------------------------------------
  // Comparison
  // -------------------------------------------------------------------------

  /**
   * Compare a new observation against the personal baseline.
   *
   * This function NEVER claims that an observation is dangerous.
   * It only answers:
   *
   * "How different is this from the user's learned normal?"
   */
  compare(
    metric: BaselineMetric,
    value: number | null | undefined,
  ): BaselineComparison {
    if (!isValidObservation(value)) {
      return {
        deviation: null,
        zScore: null,
        baselineReady: false,
        confidence: 0,
      };
    }

    const state =
      this.getMetricState(metric);

    /**
     * Cold-start guard.
     *
     * Until enough clean history exists, we explicitly refuse to
     * describe the current value as abnormal relative to this person.
     */
    if (
      !state.ready ||
      state.mean === null
    ) {
      return {
        deviation: null,
        zScore: null,
        baselineReady: false,
        confidence: 0,
      };
    }

    const deviation =
      value - state.mean;

    /**
     * A near-zero variance means we cannot safely calculate a
     * meaningful standardized deviation.
     *
     * Returning null is safer than generating an enormous artificial
     * z-score from division by an almost-zero number.
     */
    if (
      state.stdDev === null ||
      state.stdDev <= this.minimumStdDev
    ) {
      return {
        deviation,
        zScore: null,
        baselineReady: true,
        confidence: state.confidence,
      };
    }

    const zScore =
      deviation / state.stdDev;

    if (!Number.isFinite(zScore)) {
      return {
        deviation,
        zScore: null,
        baselineReady: true,
        confidence: state.confidence,
      };
    }

    return {
      deviation,
      zScore,
      baselineReady: true,
      confidence: state.confidence,
    };
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Returns whether the specified metric has completed cold start.
   */
  isReady(
    metric: BaselineMetric,
  ): boolean {
    return this.getMetricState(metric).ready;
  }

  /**
   * Returns whether all currently tracked metrics have enough
   * observations to establish a personal baseline.
   *
   * NOTE:
   * This is intentionally strict. Downstream modules should normally
   * check the specific metric they need rather than requiring every
   * sensor to be ready.
   */
  isFullyInitialized(): boolean {
    return (
      this.isReady("heartRate") &&
      this.isReady("rrInterval") &&
      this.isReady("rmssd") &&
      this.isReady("sdnn") &&
      this.isReady("temperature")
    );
  }

  /**
   * Reset every personal baseline.
   *
   * WHY:
   * A sensor/user session can be invalidated or intentionally restarted.
   * Keeping stale history after such an event could cause the new session
   * to be compared against an inappropriate baseline.
   */
  reset(): void {
    for (const metric of Object.keys(
      this.metrics,
    ) as BaselineMetric[]) {
      this.metrics[metric] =
        createMetricBuffer();
    }
  }

  /**
   * Reset one metric without destroying the other personal baselines.
   */
  resetMetric(
    metric: BaselineMetric,
  ): void {
    this.metrics[metric] =
      createMetricBuffer();
  }
}