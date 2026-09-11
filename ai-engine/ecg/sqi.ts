// ============================================================================
// ai-engine/ecg/sqi.ts
// Signal Quality Index (SQI) for filtered ECG.
//
// Purpose (Rule 4/5): downstream stages (R-peak detection, RR validation,
// HRV, rhythm anomaly) must NEVER be allowed to produce a confident result
// from a bad signal. SQI is the first gate in that chain — everything after
// this checks `sqi.label` before trusting a beat.
//
// Design: rule-based heuristics over simple statistics (math.ts), not a
// learned classifier — consistent with Phase 1 "DSP + statistics only"
// constraint, and swappable later for a trained model without changing
// this function's input/output shape (Rule 8).
// ============================================================================

import { FilteredECG } from "./filters";
import { SQI, SQILabel } from "../types";
import { mean, stdDev, clamp, rescale } from "../utils/math";

export interface SQIOptions {
  /**
   * Optional motion context (normalized magnitude from motion pipeline,
   * e.g. MotionPipelineOutput.motionLevel). When provided and above
   * threshold, the window is labeled MOTION_CORRUPTED regardless of
   * what the ECG statistics alone would suggest — motion artifact can
   * look deceptively "clean" in amplitude terms while still being
   * physiologically meaningless (Part 5.1 / Part 6.1 motion-gating).
   */
  motionLevel?: number;
}

// Expected filtered-ECG amplitude band for a well-attached lead, in the
// same normalized units the synthetic generator and real ADC front-end
// are expected to produce (R-peak amplitude normalized ~1.0). These are
// deliberately generous bounds — SQI's job is to catch clearly bad signal,
// not to be a strict morphology validator (that's rPeakDetection's job).
const FLATLINE_STDDEV_THRESHOLD = 0.01; // near-zero variance => lead off / no contact
const SATURATION_ABS_THRESHOLD = 4.5; // clipped/railed ADC or gross artifact
const SATURATION_FRACTION_LIMIT = 0.02; // >2% of samples railed => INVALID
const NOISY_STDDEV_HIGH = 0.6; // stdDev this high on a 0.5-40Hz signal is not clean ECG
const MOTION_LEVEL_THRESHOLD = 0.5; // normalized motion magnitude gate

/**
 * Zero-crossing rate: how often the signal crosses its own mean per sample.
 * A clean ECG is dominated by a slow baseline with sharp, infrequent QRS
 * spikes -> low zero-crossing rate. High-frequency noise (EMG, poor
 * contact, motion) crosses the mean far more often. Used here only as a
 * coarse "is this spectrally noisy" proxy, since a full FFT is unnecessary
 * overhead for a quality gate (Rule 7: mobile efficiency).
 */
function zeroCrossingRate(samples: number[], baseline: number): number {
  if (samples.length < 2) return 0;
  let crossings = 0;
  for (let i = 1; i < samples.length; i++) {
    const prevAbove = samples[i - 1] >= baseline;
    const currAbove = samples[i] >= baseline;
    if (prevAbove !== currAbove) crossings += 1;
  }
  return crossings / (samples.length - 1);
}

export function computeSQI(filtered: FilteredECG, options: SQIOptions = {}): SQI {
  const { filteredSamples } = filtered;

  // Guard: no samples at all is not "clean", it's invalid — never let an
  // empty window silently pass as GOOD (Rule 4).
  if (filteredSamples.length === 0) {
    return { label: "INVALID", score: 0 };
  }

  // --- Motion gate takes priority over everything else ---------------------
  // If we know the body is moving significantly, the ECG window is not
  // trustworthy regardless of how "clean" its statistics look, because
  // motion artifact can masquerade as normal-amplitude signal.
  if (options.motionLevel !== undefined && options.motionLevel >= MOTION_LEVEL_THRESHOLD) {
    return {
      label: "MOTION_CORRUPTED",
      // Score decays further as motion increases past threshold, so callers
      // doing soft weighting (fusion layer) get more than a binary signal.
      score: clamp(1 - rescale(options.motionLevel, MOTION_LEVEL_THRESHOLD, 1.0, 0, 1), 0, 0.4),
    };
  }

  const m = mean(filteredSamples) ?? 0;
  const sd = stdDev(filteredSamples) ?? 0;

  // --- Flatline / lead-off detection ---------------------------------------
  // Near-zero variance after bandpass filtering means there is no cardiac
  // signal present at all (electrode off skin, dead sensor, disconnected
  // lead) — this must be INVALID, never silently treated as "very calm ECG".
  if (sd < FLATLINE_STDDEV_THRESHOLD) {
    return { label: "INVALID", score: 0 };
  }

  // --- Saturation / clipping detection -------------------------------------
  // Large fraction of samples pinned at extreme amplitude indicates ADC
  // clipping or a gross artifact (e.g. lead yanked, static discharge),
  // not physiological signal.
  const saturatedCount = filteredSamples.filter((v) => Math.abs(v) >= SATURATION_ABS_THRESHOLD).length;
  const saturatedFraction = saturatedCount / filteredSamples.length;
  if (saturatedFraction > SATURATION_FRACTION_LIMIT) {
    return { label: "INVALID", score: 0 };
  }

  // --- Noise-energy heuristics ----------------------------------------------
  const zcr = zeroCrossingRate(filteredSamples, m);

  // Excess variance on a bandpass-filtered signal, or an unusually high
  // zero-crossing rate, both indicate residual noise the filters couldn't
  // remove (e.g. loose contact, EMG bleed-through) — not a hard invalidation,
  // but not trustworthy for confident beat detection either.
  const isHighVariance = sd > NOISY_STDDEV_HIGH;
  const isHighZCR = zcr > 0.35; // clean ECG: mostly-flat baseline, few crossings

  if (isHighVariance && isHighZCR) {
    // Both signs of noise present at once => NOISY, low score
    return { label: "NOISY", score: clamp(0.15, 0, 1) };
  }

  if (isHighVariance || isHighZCR) {
    // One sign present => USABLE (downstream stages may proceed with
    // reduced confidence, per Rule 5's confidence-threshold gate), not GOOD.
    const score = clamp(rescale(sd, NOISY_STDDEV_HIGH, NOISY_STDDEV_HIGH * 2, 0.6, 0.3), 0.3, 0.6);
    return { label: "USABLE" as SQILabel, score };
  }

  // --- Clean signal ----------------------------------------------------------
  // Score scales with how far stdDev sits inside the "clean" band — closer
  // to a typical, well-formed QRS-dominated signal scores higher, rather
  // than treating all "GOOD" windows as equally good.
  const score = clamp(rescale(sd, FLATLINE_STDDEV_THRESHOLD, NOISY_STDDEV_HIGH, 0.75, 1.0), 0.75, 1.0);
  return { label: "GOOD", score };
}