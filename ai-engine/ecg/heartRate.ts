// ============================================================================
// ai-engine/ecg/heartRate.ts
// Rolling instantaneous heart rate from VALIDATED RR intervals only.
//
// Uses median, not mean, of recent RR intervals -> HR conversion. Median is
// more robust to any single RR value that slipped past rrValidation.ts but
// is still a mild outlier (MAD rejection has a threshold, not a guarantee),
// consistent with the "robust over precise" priority for a safety-facing
// heart-rate readout (Rule 7: Reliability > False-alarm resistance > Accuracy).
// ============================================================================

import { median } from "../utils/math";

export interface HeartRateOptions {
  /** How many of the most recent validated RR intervals to use. Small
   * windows (e.g. 5-8 beats) keep HR responsive to real changes; too large
   * a window would smooth over genuine bradycardia/tachycardia onset. */
  windowSize?: number;
  /** Minimum number of validated RR intervals required before HR is
   * reported at all — below this, we don't have enough evidence for a
   * confident number (Rule 5: never decide from too little data). */
  minIntervalsRequired?: number;
}

/**
 * Computes rolling heart rate (bpm) from the most recent validated RR
 * intervals. Returns null (never 0/NaN) if there isn't enough validated
 * data yet — callers must treat null as "not ready", not "flatline".
 */
export function computeHeartRate(
  rrIntervalsMs: number[],
  options: HeartRateOptions = {}
): number | null {
  const { windowSize = 8, minIntervalsRequired = 3 } = options;

  if (rrIntervalsMs.length < minIntervalsRequired) {
    return null;
  }

  const recent = rrIntervalsMs.slice(-windowSize);
  const medianRR = median(recent);
  if (medianRR === null || medianRR <= 0) {
    return null;
  }

  return 60000 / medianRR;
}