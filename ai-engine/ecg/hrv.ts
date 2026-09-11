// ============================================================================
// ai-engine/ecg/hrv.ts
// SDNN / RMSSD / pNN50 from validated RR intervals, with a confidence score
// reflecting whether the window is long enough for that metric to be
// meaningful (Part 6.6, per Task Force of ESC/NASPE 1996 HRV standards).
//
// ESC/NASPE 1996 guidance (short-term HRV):
//   - RMSSD and pNN50 are time-domain metrics that remain reasonably
//     interpretable on short recordings (down to ~1-2 minutes of clean
//     data), since they capture beat-to-beat (vagally-mediated) variation.
//   - SDNN reflects overall variance across the whole recording and is
//     standardly reported over much longer windows (5min-24hr); computing
//     it on a short rolling window is still done in practice but is
//     LESS reliable/comparable -> we always report it, but its confidence
//     is capped lower than RMSSD/pNN50 for the same window length.
// This module never withholds a number outright once minimums are met —
// it reports the number AND a confidence, so callers (fusion/decision) can
// decide how much weight to give it, rather than us silently guessing.
// ============================================================================

import { HRVResult } from "../types";
import { sdnn as computeSDNN, rmssd as computeRMSSD, pnn50 as computePNN50 } from "../utils/math";

export interface HRVOptions {
  /** Absolute minimum validated RR intervals before ANY HRV metric is
   * computed at all — below this, even RMSSD is not meaningful. */
  minIntervalsForAnyMetric?: number;
  /** RR interval count considered "fully adequate" for RMSSD/pNN50
   * short-term confidence to reach 1.0 (rough proxy for ~1-2 min clean
   * data at typical resting HR). */
  fullConfidenceIntervalsShortTerm?: number;
  /** RR interval count considered "fully adequate" for SDNN — deliberately
   * much larger than the short-term metrics, per ESC/NASPE guidance above. */
  fullConfidenceIntervalsSDNN?: number;
}

export function computeHRV(
  rrIntervalsMs: number[],
  options: HRVOptions = {}
): HRVResult | null {
  const {
    minIntervalsForAnyMetric = 5,
    fullConfidenceIntervalsShortTerm = 40, // ~ a couple minutes at resting HR
    fullConfidenceIntervalsSDNN = 200, // ~ several minutes; SDNN confidence intentionally harder to max out
  } = options;

  if (rrIntervalsMs.length < minIntervalsForAnyMetric) {
    // Not enough beats to say anything about variability at all — return
    // null wholesale rather than a misleadingly precise-looking object.
    return null;
  }

  const sdnnValue = computeSDNN(rrIntervalsMs);
  const rmssdValue = computeRMSSD(rrIntervalsMs);
  const pnn50Value = computePNN50(rrIntervalsMs);

  // Confidence scales linearly with interval count up to the "full
  // confidence" threshold for each metric type, then caps at 1.0. RMSSD
  // and pNN50 share a short-term confidence curve (both are successive-
  // difference metrics); SDNN uses its own, slower-climbing curve.
  const shortTermConfidence = Math.min(
    1,
    rrIntervalsMs.length / fullConfidenceIntervalsShortTerm
  );
  // pnn50Confidence is the explicit field on the contract — reuse the same
  // short-term curve, since pNN50 has the same "needs successive diffs"
  // adequacy requirement as RMSSD.
  const pnn50Confidence = shortTermConfidence;

  return {
    sdnn: sdnnValue,
    rmssd: rmssdValue,
    pnn50: pnn50Value,
    pnn50Confidence,
  };
}

/**
 * Separate confidence accessor for SDNN specifically, since SDNNs
 * confidence is NOT part of the HRVResult contract (only pnn50Confidence
 * is) but callers that want to weight SDNN in fusion (Part 8) need it.
 * Exposed as a standalone helper rather than silently baked into the
 * contract, per Rule 3 ("don't invent new field names").
 */
export function sdnnConfidence(
  rrIntervalsMs: number[],
  fullConfidenceIntervalsSDNN: number = 200
): number {
  return Math.min(1, rrIntervalsMs.length / fullConfidenceIntervalsSDNN);
}