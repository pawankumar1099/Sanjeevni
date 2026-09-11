// ============================================================================
// ai-engine/ecg/rPeakDetection.ts
// Pan-Tompkins-style adaptive-threshold QRS/R-peak detector.
//
// Reference: Pan J, Tompkins WJ. "A Real-Time QRS Detection Algorithm."
// IEEE Trans Biomed Eng. 1985. Classic 4-stage pipeline: derivative ->
// squaring -> moving-window integration (MWI) -> adaptive thresholding
// with a refractory period. We run it on the already-bandpass/notch
// filtered signal from filters.ts, so we skip PT's own bandpass stage
// (already handled, avoids double-filtering artifacts).
//
// Adaptive (not fixed) threshold is required because R-peak amplitude
// varies with electrode contact, posture, and person -> a fixed threshold
// would either miss beats on a low-amplitude signal or false-trigger on
// a high-amplitude one (Rule 5: never let a single bad assumption cascade
// into a false alert).
// ============================================================================

import { FilteredECG } from "./filters";
import { RPeakResult, Timestamp } from "../types";
import { mean } from "../utils/math";

export interface RPeakDetectionOptions {
  /** Absolute timestamp (ms) of filteredSamples[0], to convert peak indices
   * into absolute Timestamps for the RPeakResult contract. */
  timestampStart: Timestamp;
  /** Minimum physiologically-plausible refractory period between beats,
   * in ms. 250ms = 240bpm ceiling, comfortably above any real max HR,
   * used purely to reject double-detection on a single wide QRS. */
  refractoryMs?: number;
}

/** Five-point derivative approximation from the original Pan-Tompkins
 * paper: emphasizes QRS slope while suppressing P/T wave and baseline. */
function derivative(samples: number[]): number[] {
  const out = new Array(samples.length).fill(0);
  for (let i = 2; i < samples.length - 2; i++) {
    out[i] =
      (-samples[i - 2] - 2 * samples[i - 1] + 2 * samples[i + 1] + samples[i + 2]) / 8;
  }
  return out;
}

function squareSignal(samples: number[]): number[] {
  return samples.map((v) => v * v);
}

/** Moving-window integration: smooths the squared derivative into a single
 * bump per QRS complex, window width ~150ms (typical QRS duration) so it
 * spans one complex without merging adjacent beats at high HR. */
function movingWindowIntegration(samples: number[], windowSize: number): number[] {
  const out = new Array(samples.length).fill(0);
  let windowSum = 0;
  for (let i = 0; i < samples.length; i++) {
    windowSum += samples[i];
    if (i >= windowSize) windowSum -= samples[i - windowSize];
    out[i] = windowSum / Math.min(i + 1, windowSize);
  }
  return out;
}

export function detectRPeaks(
  filtered: FilteredECG,
  options: RPeakDetectionOptions
): RPeakResult {
  const { filteredSamples, sampleRate } = filtered;
  const { timestampStart, refractoryMs = 250 } = options;

  if (filteredSamples.length < 10) {
    // Too short to derive/square/integrate meaningfully — never fabricate
    // a peak from an under-length window (Rule 4).
    return { peakIndices: [], peakTimestamps: [] };
  }

  const deriv = derivative(filteredSamples);
  const squared = squareSignal(deriv);
  const windowSize = Math.max(1, Math.round(0.15 * sampleRate)); // ~150ms
  const integrated = movingWindowIntegration(squared, windowSize);

  // --- Adaptive thresholding (simplified PT dual-threshold + running peak/noise estimate) ---
  // SPKI/NPKI are running estimates of signal-peak and noise-peak levels;
  // threshold sits between them, adapting as amplitude drifts over time
  // (e.g. slow changes in electrode contact) rather than needing a fixed
  // per-session calibration.
  const initialEstimateSamples = Math.min(integrated.length, Math.round(2 * sampleRate));
  const initialSegment = integrated.slice(0, initialEstimateSamples);
  const initialMean = mean(initialSegment) ?? 0;
  const initialMax = Math.max(...initialSegment, 1e-9);

  let SPKI = initialMax * 0.5; // running signal-peak estimate
  let NPKI = initialMean * 0.5; // running noise-peak estimate
  const refractorySamples = Math.round((refractoryMs / 1000) * sampleRate);

  const peakIndices: number[] = [];
  let lastPeakIdx = -refractorySamples;

  for (let i = 1; i < integrated.length - 1; i++) {
    const threshold = NPKI + 0.25 * (SPKI - NPKI);

    // Local maximum check: candidate must exceed neighbors and threshold,
    // and respect the refractory period (rejects double-triggering on one
    // wide QRS complex's rising/falling edges).
    const isLocalMax = integrated[i] > integrated[i - 1] && integrated[i] >= integrated[i + 1];
    const pastRefractory = i - lastPeakIdx >= refractorySamples;

    if (isLocalMax && integrated[i] > threshold && pastRefractory) {
      peakIndices.push(i);
      lastPeakIdx = i;
      SPKI = 0.125 * integrated[i] + 0.875 * SPKI; // signal-peak update (PT constants)
    } else if (isLocalMax) {
      NPKI = 0.125 * integrated[i] + 0.875 * NPKI; // noise-peak update
    }
  }

  // Peaks were detected on the integrated/derivative signal, which lags
  // the true R-peak by roughly the derivative+MWI group delay. Refine each
  // candidate by finding the true local maximum of the ORIGINAL filtered
  // signal within a small window around the candidate — this keeps R-peak
  // timing accurate for RR-interval math, not just detection.
  const refineWindow = Math.max(1, Math.round(0.05 * sampleRate)); // ~50ms
  const refinedIndices = peakIndices.map((idx) => {
    const lo = Math.max(0, idx - refineWindow);
    const hi = Math.min(filteredSamples.length - 1, idx + refineWindow);
    let bestIdx = idx;
    let bestVal = -Infinity;
    for (let j = lo; j <= hi; j++) {
      if (filteredSamples[j] > bestVal) {
        bestVal = filteredSamples[j];
        bestIdx = j;
      }
    }
    return bestIdx;
  });

  // Deduplicate in case refinement collapsed two close candidates onto the
  // same true peak (can happen near the refractory boundary).
  const uniqueSorted = Array.from(new Set(refinedIndices)).sort((a, b) => a - b);

  const peakTimestamps = uniqueSorted.map(
    (idx) => timestampStart + (idx / sampleRate) * 1000
  );

  return { peakIndices: uniqueSorted, peakTimestamps };
}