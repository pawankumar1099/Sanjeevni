// ============================================================================
// ai-engine/ecg/filters.ts
// Stateful streaming filters for raw ECG: 50Hz notch (powerline hum removal)
// + 0.5-40Hz bandpass (baseline wander + high-frequency EMG/noise removal).
//
// Design choice: 2nd-order IIR biquads (RBJ Audio-EQ-Cookbook formulas),
// NOT FFT-based filtering. Reasons (Priority: Reliability > Real-time perf >
// Mobile efficiency, per project priority order):
//   - O(1) time and O(1) memory per sample -> safe on-device, no buffering
//     delay, no block-boundary artifacts.
//   - Naturally streaming: filters carry state (x[n-1], x[n-2], y[n-1], y[n-2])
//     across process() calls, so a live BLE stream can be filtered chunk by
//     chunk without discontinuities at chunk boundaries.
//   - FFT filtering requires fixed-size windows and reintroduces the exact
//     "buffer before we can act" latency this device is trying to avoid
//     (Part 6.1 startup-transient concerns apply here too).
//
// Bandwidth choice: 0.5-40Hz is the standard bandwidth for AMBULATORY /
// RHYTHM-MONITORING ECG (as opposed to 0.05-150Hz diagnostic 12-lead ECG
// per AHA/ACC/HRS 2007 recommendations for diagnostic ECG equipment).
// Sanjeevni is explicitly a wearable rhythm/risk monitor, not a diagnostic
// 12-lead system, so the tighter ambulatory bandwidth is the scientifically
// appropriate choice: it removes baseline wander (<0.5Hz, from respiration
// and motion) and EMG/muscle noise (>40Hz) while preserving QRS morphology
// needed for R-peak detection.
//
// Notch choice: 50Hz for Indian mains frequency (Sanjeevni targets India /
// SIH 2026). A narrow-Q notch (Q=30) is used so only the powerline tone is
// attenuated, not the neighboring QRS spectral energy.
// ============================================================================

import { ECGSample } from "../types";

export interface FilteredECG {
  filteredSamples: number[];
  sampleRate: number;
}

// ---------------------------------------------------------------------------
// Generic stateful Biquad IIR filter (Direct Form I)
// ---------------------------------------------------------------------------

type BiquadType = "lowpass" | "highpass" | "notch";

class Biquad {
  // Filter coefficients (normalized so a0 = 1)
  private b0 = 1;
  private b1 = 0;
  private b2 = 0;
  private a1 = 0;
  private a2 = 0;

  // State: previous 2 inputs and previous 2 outputs (Direct Form I).
  // Direct Form I is used over Direct Form II because it's numerically
  // more robust to coefficient quantization at low frequencies (our
  // 0.5Hz high-pass cutoff is very close to DC relative to typical
  // 250-360Hz ECG sample rates), which matters more here than the
  // slightly smaller memory footprint of Direct Form II.
  private x1 = 0;
  private x2 = 0;
  private y1 = 0;
  private y2 = 0;

  constructor(type: BiquadType, sampleRate: number, freq: number, Q: number) {
    this.configure(type, sampleRate, freq, Q);
  }

  /**
   * (Re)configures coefficients. Called on construction, and again if the
   * incoming sample rate changes mid-stream (e.g. sensor reconnects at a
   * different rate) — state is intentionally NOT reset here, only
   * coefficients, since configure() is always followed by reset() at the
   * call sites that need a clean state.
   */
  configure(type: BiquadType, sampleRate: number, freq: number, Q: number): void {
    const w0 = (2 * Math.PI * freq) / sampleRate;
    const cosW0 = Math.cos(w0);
    const sinW0 = Math.sin(w0);
    const alpha = sinW0 / (2 * Q);

    let b0: number, b1: number, b2: number, a0: number, a1: number, a2: number;

    switch (type) {
      case "lowpass":
        b0 = (1 - cosW0) / 2;
        b1 = 1 - cosW0;
        b2 = (1 - cosW0) / 2;
        a0 = 1 + alpha;
        a1 = -2 * cosW0;
        a2 = 1 - alpha;
        break;

      case "highpass":
        b0 = (1 + cosW0) / 2;
        b1 = -(1 + cosW0);
        b2 = (1 + cosW0) / 2;
        a0 = 1 + alpha;
        a1 = -2 * cosW0;
        a2 = 1 - alpha;
        break;

      case "notch":
        b0 = 1;
        b1 = -2 * cosW0;
        b2 = 1;
        a0 = 1 + alpha;
        a1 = -2 * cosW0;
        a2 = 1 - alpha;
        break;
    }

    // Normalize by a0 so process() never needs to divide per-sample
    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
  }

  /** Clears filter memory. Used when a sensor disconnects/reconnects
   * (matching RingBuffer.clear() semantics elsewhere in the engine) so
   * stale state doesn't bleed into a fresh session. */
  reset(): void {
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }

  /** Process a single sample, updating internal state. */
  processSample(x0: number): number {
    const y0 =
      this.b0 * x0 + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;

    this.x2 = this.x1;
    this.x1 = x0;
    this.y2 = this.y1;
    this.y1 = y0;

    return y0;
  }

  /** Process a chunk, preserving state across calls (streaming). */
  processArray(samples: number[]): number[] {
    const out = new Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      out[i] = this.processSample(samples[i]);
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// ECGFilterChain — the public, stateful filter pipeline used by callers.
// Cascades: high-pass (0.5Hz) -> low-pass (40Hz) -> notch (50Hz).
// ---------------------------------------------------------------------------

export class ECGFilterChain {
  private highpass: Biquad | null = null;
  private lowpass: Biquad | null = null;
  private notch: Biquad | null = null;
  private configuredSampleRate: number | null = null;

  // Q values chosen deliberately, not defaults:
  // - Highpass/lowpass Q = 0.707 (Butterworth, maximally flat passband) —
  //   the standard choice for ECG conditioning since it avoids introducing
  //   passband ripple that could distort QRS amplitude/morphology and
  //   confuse downstream R-peak detection.
  // - Notch Q = 30 — narrow enough to spare the ECG spectrum near 50Hz,
  //   wide enough to tolerate small mains-frequency drift.
  private static readonly BUTTERWORTH_Q = 0.70710678; // 1/sqrt(2)
  private static readonly NOTCH_Q = 30;
  private static readonly HIGHPASS_HZ = 0.5;
  private static readonly LOWPASS_HZ = 40;
  private static readonly NOTCH_HZ = 50; // Indian mains frequency

  /**
   * (Re)builds the filter stages if this is the first call, or if the
   * incoming sample rate has changed since the last chunk (e.g. sensor
   * reconnect at a different rate). A genuine sample-rate change also
   * forces a state reset, since old filter memory computed at the old
   * rate is not meaningfully continuous with the new rate.
   */
  private ensureConfigured(sampleRate: number): void {
    if (this.configuredSampleRate === sampleRate && this.highpass && this.lowpass && this.notch) {
      return;
    }

    this.highpass = new Biquad("highpass", sampleRate, ECGFilterChain.HIGHPASS_HZ, ECGFilterChain.BUTTERWORTH_Q);
    this.lowpass = new Biquad("lowpass", sampleRate, ECGFilterChain.LOWPASS_HZ, ECGFilterChain.BUTTERWORTH_Q);
    this.notch = new Biquad("notch", sampleRate, ECGFilterChain.NOTCH_HZ, ECGFilterChain.NOTCH_Q);
    this.configuredSampleRate = sampleRate;
  }

  /**
   * Filters one incoming ECG chunk, preserving filter state from any
   * previous chunk at the same sample rate. This is the function pipeline
   * callers (e.g. rPeakDetection.ts) invoke per BLE tick.
   */
  process(ecg: ECGSample): FilteredECG {
    this.ensureConfigured(ecg.sampleRate);

    // Cascade order matters: remove baseline wander first (highpass),
    // then high-frequency noise (lowpass), then the narrow powerline
    // tone (notch) — this ordering avoids the notch fighting against
    // a still-present, slowly-drifting DC/baseline offset.
    let stage = this.highpass!.processArray(ecg.samples);
    stage = this.lowpass!.processArray(stage);
    stage = this.notch!.processArray(stage);

    return {
      filteredSamples: stage,
      sampleRate: ecg.sampleRate,
    };
  }

  /** Clears all filter memory — call when a sensor disconnects/reconnects. */
  reset(): void {
    this.highpass?.reset();
    this.lowpass?.reset();
    this.notch?.reset();
  }
}