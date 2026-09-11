// ============================================================================
// ai-engine/ecg/rrValidation.ts
// Converts detected R-peak timestamps into validated RR intervals.
//
// Two-stage validation (Part 6.5):
//   1. Physiological plausibility bounds — reject intervals outside what
//      a human heart can produce, independent of any history.
//   2. MAD-based outlier rejection against recent accepted RR values —
//      catches a single spurious beat (missed or extra detection) that is
//      individually "plausible" in isolation but discontinuous from the
//      person's own recent rhythm.
// Never trust a single RR value to drive HR/HRV without this gate — one
// missed/extra beat can otherwise double or halve an HR reading (Rule 5).
// ============================================================================

import { RPeakResult, RRResult } from "../types";
import { isOutlierByMAD } from "../utils/math";

export interface RRValidationOptions {
  /** Plausibility bounds in ms. Default 300ms (200bpm) to 2000ms (30bpm)
   * covers rest through vigorous exercise/bradycardia with margin, without
   * being so loose it admits detector errors. */
  minRRMs?: number;
  maxRRMs?: number;
  /** How many recent accepted RR values to compare a candidate against
   * for MAD outlier rejection (Part 6.5: "previous 3-5 accepted values"). */
  madWindowSize?: number;
  madThreshold?: number;
}

export function validateRRIntervals(
  rPeaks: RPeakResult,
  options: RRValidationOptions = {}
): RRResult {
  const {
    minRRMs = 300,
    maxRRMs = 2000,
    madWindowSize = 5,
    madThreshold = 3.5,
  } = options;

  const timestamps = rPeaks.peakTimestamps;
  if (timestamps.length < 2) {
    // Can't form an RR interval from fewer than 2 peaks — return empty,
    // never fabricate an interval (Rule 4).
    return { rrIntervalsMs: [], rejectedCount: 0 };
  }

  const accepted: number[] = [];
  let rejectedCount = 0;

  for (let i = 1; i < timestamps.length; i++) {
    const rr = timestamps[i] - timestamps[i - 1];

    // Stage 1: absolute plausibility bounds.
    if (rr < minRRMs || rr > maxRRMs) {
      rejectedCount += 1;
      continue;
    }

    // Stage 2: MAD outlier check against the most recent accepted values.
    // Only applied once we have enough accepted history to compare against
    // — with fewer than 2 prior accepted values, MAD isn't meaningful, so
    // we fall through to plausibility-only acceptance for those first beats.
    const recentAccepted = accepted.slice(-madWindowSize);
    if (recentAccepted.length >= 2 && isOutlierByMAD(rr, recentAccepted, madThreshold)) {
      rejectedCount += 1;
      continue;
    }

    accepted.push(rr);
  }

  return { rrIntervalsMs: accepted, rejectedCount };
}