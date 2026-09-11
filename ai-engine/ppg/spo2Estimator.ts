// ============================================================================
// ai-engine/ppg/spo2Estimator.ts
// SpO2 estimation via the standard ratio-of-ratios (R) method, with an
// empirical calibration curve, PLUS a PPG-derived HR as a cross-check
// value only (per types.ts contract: "cross-check only, never primary" —
// heartRate.ts's ECG-derived HR remains the source of truth).
//
// Method (standard pulse oximetry physics, e.g. Webster JG, "Design of
// Pulse Oximeters", 1997 — widely-cited reference textbook for this exact
// R -> SpO2 empirical calibration approach used across commercial and
// research-grade pulse oximeters):
//   R = (AC_red / DC_red) / (AC_ir / DC_ir)
//   SpO2 (%) = A - B * R   [linear empirical approximation, valid ~70-100%]
// True commercial devices calibrate A/B against human trials across many
// units; we use widely-published approximate constants (A=110, B=25) as a
// Phase 1 stand-in — NOT clinically calibrated, consistent with Rule 6
// (risk-oriented device, not diagnostic-grade SpO2).
// ============================================================================

import { PPGSample, PPGPipelineOutput } from "../types";
import { mean, stdDev, clamp } from "../utils/math";
import { assessPPGQuality, PPGQualityOptions } from "./qualityGate";

// Empirical linear calibration constants (A - B*R form), within the
// commonly-published range for low-cost reflective/transmissive PPG
// pulse oximetry approximations. Explicitly NOT a substitute for a
// clinically validated calibration (Rule 9: cite real, don't fabricate
// precision this build hasn't earned).
const CALIBRATION_A = 110;
const CALIBRATION_B = 25;

// Physiologically valid SpO2 output range — even if the linear formula
// mathematically produces a value outside this, we clamp rather than
// report an impossible reading (Rule 4/6).
const SPO2_MIN = 70;
const SPO2_MAX = 100;

export interface SpO2EstimationOptions extends PPGQualityOptions {}

/**
 * Simple peak-to-peak-based AC amplitude estimate for a PPG channel.
 * Used instead of full pulse-waveform peak detection (Phase 2 scope) —
 * for the ratio-of-ratios formula, a robust AC amplitude proxy across the
 * window is sufficient; we don't need per-beat waveform morphology here.
 */
function acAmplitude(samples: number[]): number {
  // stdDev-based AC estimate is more robust to a single spurious sample
  // than raw max-min peak-to-peak would be — consistent with the
  // "robust over precise" preference used elsewhere (e.g. heartRate.ts).
  return (stdDev(samples) ?? 0) * Math.SQRT2; // scale factor: approx RMS->peak for near-sinusoidal pulse waveform
}

/**
 * Very lightweight PPG-derived HR estimate via zero-crossing rate of the
 * IR channel around its mean — intentionally simple, since this value is
 * a cross-check only (types.ts: ppgDerivedHR "cross-check only, never
 * primary") and must never be mistaken for the authoritative ECG-derived
 * heart rate.
 */
function estimatePPGDerivedHR(irSamples: number[], sampleRate: number): number | null {
  if (irSamples.length < sampleRate) return null; // need at least ~1s of data

  const m = mean(irSamples) ?? 0;
  let crossings = 0;
  for (let i = 1; i < irSamples.length; i++) {
    const prevAbove = irSamples[i - 1] >= m;
    const currAbove = irSamples[i] >= m;
    if (!prevAbove && currAbove) crossings += 1; // count rising crossings only -> ~1 per beat
  }

  const durationSeconds = irSamples.length / sampleRate;
  if (durationSeconds <= 0 || crossings === 0) return null;

  const bpm = (crossings / durationSeconds) * 60;

  // Sanity-bound: never report a cross-check HR outside plausible human
  // range — an out-of-range result means the zero-crossing proxy failed,
  // not that the person's heart rate is actually implausible.
  if (bpm < 30 || bpm > 240) return null;

  return bpm;
}

export function estimateSpO2(
  ppg: PPGSample,
  options: SpO2EstimationOptions = {}
): PPGPipelineOutput {
  const qualityResult = assessPPGQuality(ppg, options);

  // Gate: never run ratio-of-ratios math on a window the quality gate has
  // already flagged as unusable — a bad ratio computed on noise can look
  // numerically "valid" (e.g. 95%) while being physiologically meaningless
  // (Rule 5's signal-quality-gate-first ordering, mirrored from ECG).
  if (qualityResult.quality !== "GOOD") {
    return {
      spo2: null,
      spo2Confidence: 0,
      ppgQuality: qualityResult.quality,
      ppgDerivedHR: null,
    };
  }

  const { irSamples, redSamples, sampleRate } = ppg;

  const dcIR = mean(irSamples) ?? 0;
  const dcRed = mean(redSamples) ?? 0;
  const acIR = acAmplitude(irSamples);
  const acRed = acAmplitude(redSamples);

  // Guard against divide-by-zero even though the quality gate should have
  // already filtered these cases out — defense in depth, never let a
  // formula silently divide by zero and produce NaN/Infinity (Rule 4).
  if (dcIR === 0 || dcRed === 0 || acIR === 0) {
    return { spo2: null, spo2Confidence: 0, ppgQuality: "POOR", ppgDerivedHR: null };
  }

  const R = (acRed / dcRed) / (acIR / dcIR);

  const rawSpo2 = CALIBRATION_A - CALIBRATION_B * R;
  const spo2 = clamp(rawSpo2, SPO2_MIN, SPO2_MAX);

  // If clamping had to correct a wildly out-of-range raw value, that's a
  // sign the R ratio itself was unreliable even though the quality gate
  // passed — reduce confidence accordingly rather than reporting a clamped
  // number at full confidence (Rule 4: never let a corrected value look
  // as trustworthy as an unclamped one).
  const wasClamped = rawSpo2 < SPO2_MIN || rawSpo2 > SPO2_MAX;

  // Confidence combines the quality-gate's graded perfusion score with a
  // penalty if clamping was needed. This is intentionally conservative —
  // per Rule 9, we're using published approximate constants, not a
  // clinically fitted calibration, so confidence should never reach a
  // false sense of clinical precision.
  const baseConfidence = qualityResult.score;
  const spo2Confidence = clamp(wasClamped ? baseConfidence * 0.5 : baseConfidence * 0.85, 0, 0.9);

  const ppgDerivedHR = estimatePPGDerivedHR(irSamples, sampleRate);

  return {
    spo2,
    spo2Confidence,
    ppgQuality: qualityResult.quality,
    ppgDerivedHR,
  };
}