// ============================================================================
// ai-engine/utils/math.ts
// Shared numeric/statistical helpers used across the engine.
// Kept dependency-free (no external math libraries) so the whole ai-engine
// folder stays lightweight and has zero install footprint on mobile.
// ============================================================================

/**
 * Arithmetic mean. Returns null for an empty array rather than NaN,
 * so callers are forced to explicitly handle "no data" instead of
 * silently propagating NaN through downstream calculations.
 */
export function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  const sum = values.reduce((acc, v) => acc + v, 0);
  return sum / values.length;
}

/**
 * Population variance (not sample variance) — used for baseline modeling
 * where we treat the rolling window as the full population of interest.
 */
export function variance(values: number[]): number | null {
  if (values.length === 0) return null;
  const m = mean(values);
  if (m === null) return null;
  const squaredDiffs = values.map((v) => (v - m) ** 2);
  return mean(squaredDiffs);
}

/**
 * Standard deviation — sqrt of variance. Returns null if variance is null.
 */
export function stdDev(values: number[]): number | null {
  const v = variance(values);
  if (v === null) return null;
  return Math.sqrt(v);
}

/**
 * Median — used for robust outlier rejection (Part 6.5 RR validation),
 * which is intentionally more robust to a single wild value than mean.
 */
export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * SDNN — standard deviation of NN (RR) intervals.
 * Direct alias over stdDev, kept as its own named function so call
 * sites in ecg/hrv.ts read clearly and match clinical HRV terminology
 * (Task Force of ESC/NASPE, 1996 — see Part 13 references).
 */
export function sdnn(rrIntervalsMs: number[]): number | null {
  return stdDev(rrIntervalsMs);
}

/**
 * RMSSD — root mean square of successive differences between NN intervals.
 * Standard short-term HRV metric, valid even on short rolling windows
 * (Part 6.6) unlike frequency-domain metrics.
 */
export function rmssd(rrIntervalsMs: number[]): number | null {
  if (rrIntervalsMs.length < 2) return null;
  const successiveDiffsSquared: number[] = [];
  for (let i = 1; i < rrIntervalsMs.length; i++) {
    const diff = rrIntervalsMs[i] - rrIntervalsMs[i - 1];
    successiveDiffsSquared.push(diff * diff);
  }
  const meanSquaredDiff = mean(successiveDiffsSquared);
  if (meanSquaredDiff === null) return null;
  return Math.sqrt(meanSquaredDiff);
}

/**
 * pNN50 — percentage of successive RR interval differences greater than 50ms.
 * Flagged by caller as low-confidence if the window is too short
 * (Part 6.6) — this function just computes the raw value.
 */
export function pnn50(rrIntervalsMs: number[]): number | null {
  if (rrIntervalsMs.length < 2) return null;
  let countOver50 = 0;
  for (let i = 1; i < rrIntervalsMs.length; i++) {
    if (Math.abs(rrIntervalsMs[i] - rrIntervalsMs[i - 1]) > 50) {
      countOver50 += 1;
    }
  }
  return (countOver50 / (rrIntervalsMs.length - 1)) * 100;
}

/**
 * Z-score of a single value against a baseline mean/stdDev.
 * Core building block for anomaly detection (Part 6.7, Part 23) —
 * used to answer "how many standard deviations is this value from
 * this user's own personal baseline?"
 *
 * Returns null if stdDevValue is 0 or null (avoids divide-by-zero,
 * which would otherwise fabricate an infinite/undefined anomaly score).
 */
export function zScore(
  value: number,
  baselineMean: number,
  baselineStdDev: number
): number | null {
  if (!baselineStdDev || baselineStdDev === 0) return null;
  return (value - baselineMean) / baselineStdDev;
}

/**
 * Median Absolute Deviation-based outlier check.
 * More robust than a simple z-score for rejecting a single wild RR
 * interval, since MAD is not itself skewed by the outlier the way
 * mean/stdDev can be (Part 6.5: "not wildly discontinuous from
 * previous 3-5 accepted RR values").
 */
export function isOutlierByMAD(
  value: number,
  recentValues: number[],
  threshold: number = 3.5
): boolean {
  if (recentValues.length === 0) return false;
  const med = median(recentValues);
  if (med === null) return false;
  const absoluteDeviations = recentValues.map((v) => Math.abs(v - med));
  const mad = median(absoluteDeviations);
  if (mad === null || mad === 0) return false;
  // 0.6745 is the standard consistency constant for MAD under normality
  const modifiedZ = (0.6745 * (value - med)) / mad;
  return Math.abs(modifiedZ) > threshold;
}

/**
 * Clamp a value between min/max — used everywhere confidence scores
 * (0.0-1.0) must never spill outside valid bounds due to floating-point
 * or formula edge cases.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Linear rescale of a value from one range to another, clamped to the
 * output range. Used repeatedly to turn raw physical quantities (e.g.
 * heat index in °C) into normalized 0-1 risk scores.
 */
export function rescale(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  if (inMax === inMin) return outMin; // avoid divide-by-zero
  const t = (value - inMin) / (inMax - inMin);
  return clamp(outMin + t * (outMax - outMin), outMin, outMax);
}

/**
 * Exponentially weighted moving average update — a single-step,
 * O(1)-memory way to maintain a rolling baseline without storing
 * full history (used by baseline/personalBaseline.ts).
 */
export function ewmaUpdate(
  previousEwma: number | null,
  newValue: number,
  alpha: number = 0.1
): number {
  if (previousEwma === null) return newValue;
  return alpha * newValue + (1 - alpha) * previousEwma;
}