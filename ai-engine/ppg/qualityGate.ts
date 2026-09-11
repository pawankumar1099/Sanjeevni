// ============================================================================
// ai-engine/ppg/qualityGate.ts
// PPG signal quality / validity gate — the PPG-pipeline equivalent of
// ecg/sqi.ts. Runs BEFORE spo2Estimator.ts, since ratio-of-ratios SpO2 math
// is extremely sensitive to poor perfusion, motion artifact, and ambient
// light/pressure-induced saturation (Rule 5: never estimate SpO2 from a
// window we already know is unreliable).
//
// PPG-specific failure modes this gate targets (distinct from ECG's):
//   - Low perfusion / poor contact: AC amplitude too small relative to DC
//     to yield a stable ratio (classic "finger not seated" case).
//   - Motion artifact: PPG is FAR more motion-sensitive than ECG since it
//     relies on blood-volume-driven light absorption changes, which a
//     moving limb swamps with mechanical artifact.
//   - Saturation/clipping: ambient light leakage or pressed-too-hard
//     sensor placement can rail the photodiode reading.
// ============================================================================

import { PPGSample } from "../types";
import { mean, stdDev, clamp, rescale } from "../utils/math";

export type PPGQuality = "GOOD" | "POOR" | "UNAVAILABLE";

export interface PPGQualityResult {
  quality: PPGQuality;
  score: number; // 0.0-1.0, graded like SQI.score, for fusion weighting
  reasons: string[];
}

export interface PPGQualityOptions {
  /** Same motion-context gate concept as ecg/sqi.ts — PPG is even more
   * motion-sensitive than ECG, so this threshold is intentionally lower
   * than the ECG motion gate's. */
  motionLevel?: number;
}

// Thresholds are on raw ADC-style units matching what a real MAX30102-class
// sensor (typical low-cost PPG front-end used in these builds) and the
// synthetic generator are expected to produce. Deliberately generous —
// this gate's job is to catch clearly-bad windows, not perform amplitude
// calibration (that's implicitly handled by the ratio-of-ratios math itself).
const MIN_SAMPLES = 10;
const FLATLINE_STDDEV_FRACTION = 0.005; // AC component < 0.5% of DC mean => no pulsatile signal
const SATURATION_FRACTION_LIMIT = 0.02;
const MOTION_LEVEL_THRESHOLD = 0.35; // lower than ECG's 0.5 — PPG degrades sooner under motion
const SATURATION_HIGH = 262143; // 18-bit ADC max, typical MAX30102-class ceiling
const SATURATION_LOW = 0;

function saturationFraction(samples: number[]): number {
  if (samples.length === 0) return 0;
  const saturated = samples.filter((v) => v <= SATURATION_LOW || v >= SATURATION_HIGH).length;
  return saturated / samples.length;
}

export function assessPPGQuality(
  ppg: PPGSample,
  options: PPGQualityOptions = {}
): PPGQualityResult {
  const { irSamples, redSamples } = ppg;
  const reasons: string[] = [];

  // Guard: missing/too-short data is UNAVAILABLE, never treated as POOR-
  // but-present (Rule 4: unavailable must never look like degraded data,
  // it must look like absent data).
  if (
    !irSamples ||
    !redSamples ||
    irSamples.length < MIN_SAMPLES ||
    redSamples.length < MIN_SAMPLES ||
    irSamples.length !== redSamples.length
  ) {
    return { quality: "UNAVAILABLE", score: 0, reasons: ["Insufficient or mismatched IR/red sample data"] };
  }

  // --- Motion gate first, same priority ordering as ECG SQI ----------------
  if (options.motionLevel !== undefined && options.motionLevel >= MOTION_LEVEL_THRESHOLD) {
    return {
      quality: "POOR",
      score: clamp(1 - rescale(options.motionLevel, MOTION_LEVEL_THRESHOLD, 1.0, 0, 1), 0, 0.3),
      reasons: ["Motion level too high for reliable PPG"],
    };
  }

  // --- Saturation / clipping -------------------------------------------------
  const irSat = saturationFraction(irSamples);
  const redSat = saturationFraction(redSamples);
  if (irSat > SATURATION_FRACTION_LIMIT || redSat > SATURATION_FRACTION_LIMIT) {
    return { quality: "POOR", score: 0.1, reasons: ["IR or red channel saturated/clipped"] };
  }

  // --- Perfusion adequacy: AC (pulsatile) amplitude relative to DC mean ----
  // A real finger with good contact produces a clear pulsatile AC swing on
  // top of a large DC offset. Near-flat AC relative to DC means no usable
  // pulse waveform — lead-off or extremely poor contact, analogous to
  // ECG's flatline/lead-off check.
  const irMean = mean(irSamples) ?? 0;
  const irStdDev = stdDev(irSamples) ?? 0;
  const redMean = mean(redSamples) ?? 0;
  const redStdDev = stdDev(redSamples) ?? 0;

  if (irMean <= 0 || redMean <= 0) {
    return { quality: "UNAVAILABLE", score: 0, reasons: ["Non-physical zero/negative DC baseline"] };
  }

  const irACFraction = irStdDev / irMean;
  const redACFraction = redStdDev / redMean;

  if (irACFraction < FLATLINE_STDDEV_FRACTION || redACFraction < FLATLINE_STDDEV_FRACTION) {
    reasons.push("Pulsatile (AC) component too small relative to DC baseline — likely poor contact");
    return { quality: "POOR", score: 0.15, reasons };
  }

  // --- Clean window ----------------------------------------------------------
  // Score scales with how comfortably the AC fraction clears the flatline
  // floor, capped — mirrors ecg/sqi.ts's graded-score approach so fusion
  // gets more than a binary signal.
  const perfusionScore = clamp(
    rescale(Math.min(irACFraction, redACFraction), FLATLINE_STDDEV_FRACTION, FLATLINE_STDDEV_FRACTION * 6, 0.6, 1.0),
    0.6,
    1.0
  );

  reasons.push("Adequate pulsatile signal on both IR and red channels");
  return { quality: "GOOD", score: perfusionScore, reasons };
}