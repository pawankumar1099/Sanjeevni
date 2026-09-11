// ============================================================================
// ai-engine/mock/syntheticDataGenerator.ts
// Generates synthetic sensor data for development and testing, completely
// decoupled from real hardware. This unblocks Hour 0-1 of the build plan
// (Part 14) and is also reused by __tests__/ for repeatable test fixtures.
//
// IMPORTANT: this file is dev/test tooling only — it must never be imported
// by any production pipeline file (ecg/, motion/, environment/, etc).
// ============================================================================

import { ECGSample, MotionSample, EnvironmentSample } from "../types";

// ---------------------------------------------------------------------------
// ECG synthetic generator
// ---------------------------------------------------------------------------

/**
 * Generates a synthetic ECG-like waveform using a simplified sum-of-Gaussians
 * model for the QRS complex, repeated at a target heart rate, with optional
 * noise injection. This is NOT a clinically accurate ECG morphology model —
 * it's a lightweight stand-in good enough to validate filtering, R-peak
 * detection, and RR/HR/HRV math without needing a real dataset for Phase 1
 * scaffolding. (Real MIT-BIH data should be used later for Phase 2 model
 * validation — see Part 35.)
 */
export function generateSyntheticECG(options: {
  durationSeconds: number;
  sampleRate?: number; // default 300 Hz, within the 250-360 Hz spec
  heartRateBpm?: number; // default 72
  noiseLevel?: number; // 0 = clean, higher = noisier baseline
  powerlineHumAmplitude?: number; // simulates 50Hz interference
  motionArtifact?: boolean; // injects a burst of large-amplitude motion noise
  timestampStart?: number;
}): ECGSample {
  const {
    durationSeconds,
    sampleRate = 300,
    heartRateBpm = 72,
    noiseLevel = 0.02,
    powerlineHumAmplitude = 0,
    motionArtifact = false,
    timestampStart = Date.now(),
  } = options;

  const totalSamples = Math.floor(durationSeconds * sampleRate);
  const samples: number[] = new Array(totalSamples).fill(0);
  const rrIntervalSeconds = 60 / heartRateBpm;

  // Place a simplified QRS "spike" (Gaussian bump) at each expected beat time
  for (let t = rrIntervalSeconds; t < durationSeconds; t += rrIntervalSeconds) {
    const centerSampleIdx = Math.floor(t * sampleRate);
    const qrsWidthSamples = Math.floor(0.02 * sampleRate); // ~20ms QRS width
    for (
      let i = Math.max(0, centerSampleIdx - qrsWidthSamples * 3);
      i < Math.min(totalSamples, centerSampleIdx + qrsWidthSamples * 3);
      i++
    ) {
      const distance = i - centerSampleIdx;
      const gaussian = Math.exp(-(distance ** 2) / (2 * qrsWidthSamples ** 2));
      samples[i] += gaussian * 1.0; // amplitude of R-peak normalized to 1.0
    }
  }

  // Add baseline Gaussian noise
  for (let i = 0; i < totalSamples; i++) {
    samples[i] += (Math.random() - 0.5) * 2 * noiseLevel;
  }

  // Add simulated 50Hz powerline hum
  if (powerlineHumAmplitude > 0) {
    for (let i = 0; i < totalSamples; i++) {
      const timeSec = i / sampleRate;
      samples[i] += powerlineHumAmplitude * Math.sin(2 * Math.PI * 50 * timeSec);
    }
  }

  // Add a motion-artifact burst in the middle third of the signal
  if (motionArtifact) {
    const burstStart = Math.floor(totalSamples * 0.4);
    const burstEnd = Math.floor(totalSamples * 0.6);
    for (let i = burstStart; i < burstEnd; i++) {
      samples[i] += (Math.random() - 0.5) * 3.0; // large-amplitude disruption
    }
  }

  return {
    samples,
    sampleRate,
    timestampStart,
    timestampEnd: timestampStart + durationSeconds * 1000,
  };
}

// ---------------------------------------------------------------------------
// Motion synthetic generator
// ---------------------------------------------------------------------------

/**
 * Generates synthetic accelerometer/gyro data for a given activity pattern.
 * Supports the specific "fall signature" (free-fall dip -> impact spike ->
 * stillness) described in Part 5.3 / Part 18, so fallDetection.ts can be
 * tested against a known-good synthetic fall before any real hardware exists.
 */
export function generateSyntheticMotion(options: {
  durationSeconds: number;
  sampleRate?: number; // default 50 Hz
  pattern: "REST" | "WALK" | "RUN" | "FALL" | "SIT_DOWN_FAST" | "JUMP";
  timestamp?: number;
}): MotionSample {
  const { durationSeconds, sampleRate = 50, pattern, timestamp = Date.now() } = options;
  const totalSamples = Math.floor(durationSeconds * sampleRate);

  const accel: { x: number; y: number; z: number }[] = [];
  const gyro: { x: number; y: number; z: number }[] = [];

  for (let i = 0; i < totalSamples; i++) {
    const t = i / totalSamples;
    let magnitudeG = 1.0; // resting = ~1g due to gravity
    let gyroNoise = 0.02;

    switch (pattern) {
      case "REST":
        magnitudeG = 1.0 + (Math.random() - 0.5) * 0.02;
        break;

      case "WALK":
        magnitudeG = 1.0 + Math.sin(t * 40) * 0.3 + (Math.random() - 0.5) * 0.1;
        gyroNoise = 0.1;
        break;

      case "RUN":
        magnitudeG = 1.0 + Math.sin(t * 80) * 0.7 + (Math.random() - 0.5) * 0.2;
        gyroNoise = 0.3;
        break;

      case "JUMP":
        // Brief impact spike without a preceding free-fall dip —
        // used to verify the fall detector correctly REJECTS this.
        magnitudeG = t > 0.45 && t < 0.55 ? 2.5 : 1.0 + (Math.random() - 0.5) * 0.1;
        break;

      case "SIT_DOWN_FAST":
        // Moderate impact without free-fall preceding it —
        // another true-negative case for the fall detector.
        magnitudeG = t > 0.5 && t < 0.55 ? 1.8 : 1.0 + (Math.random() - 0.5) * 0.05;
        break;

      case "FALL":
        // The canonical sequence from Part 18:
        // free-fall dip -> impact spike -> post-impact stillness
        if (t > 0.3 && t < 0.4) {
          magnitudeG = 0.1; // near-zero g during free-fall
        } else if (t >= 0.4 && t < 0.42) {
          magnitudeG = 3.0; // sharp impact spike
        } else if (t >= 0.42) {
          magnitudeG = 1.0 + (Math.random() - 0.5) * 0.01; // stillness after impact
        } else {
          magnitudeG = 1.0 + (Math.random() - 0.5) * 0.05; // normal motion before fall
        }
        break;
    }

    // Distribute magnitude arbitrarily across axes (good enough for
    // magnitude-based detection logic, not meant to simulate true orientation)
    accel.push({
      x: magnitudeG * 0.33,
      y: magnitudeG * 0.33,
      z: magnitudeG * 0.34,
    });
    gyro.push({
      x: (Math.random() - 0.5) * gyroNoise,
      y: (Math.random() - 0.5) * gyroNoise,
      z: (Math.random() - 0.5) * gyroNoise,
    });
  }

  return { accel, gyro, sampleRate, timestamp };
}

// ---------------------------------------------------------------------------
// Environment synthetic generator
// ---------------------------------------------------------------------------

/**
 * Generates synthetic ambient environment readings, including named presets
 * for scenario testing (e.g. "heatwave", "normal", "poor_air_quality") that
 * map directly to the test cases required in Part 34.
 */
export function generateSyntheticEnvironment(options: {
  scenario: "NORMAL" | "HEATWAVE" | "HIGH_HUMIDITY" | "POOR_AIR_QUALITY" | "OFFLINE_NO_API";
  timestamp?: number;
}): EnvironmentSample {
  const { scenario, timestamp = Date.now() } = options;

  const presets: Record<string, EnvironmentSample> = {
    NORMAL: {
      tempC: 27,
      humidityPct: 50,
      mq135Raw: 120,
      apiAQI: 60,
      apiWeather: { tempC: 27, humidityPct: 50 },
      timestamp,
    },
    HEATWAVE: {
      tempC: 44,
      humidityPct: 35,
      mq135Raw: 150,
      apiAQI: 90,
      apiWeather: { tempC: 45, humidityPct: 32 },
      timestamp,
    },
    HIGH_HUMIDITY: {
      tempC: 33,
      humidityPct: 85,
      mq135Raw: 130,
      apiAQI: 70,
      apiWeather: { tempC: 33, humidityPct: 88 },
      timestamp,
    },
    POOR_AIR_QUALITY: {
      tempC: 30,
      humidityPct: 55,
      mq135Raw: 480,
      apiAQI: 210,
      apiWeather: { tempC: 30, humidityPct: 55 },
      timestamp,
    },
    OFFLINE_NO_API: {
      tempC: 36,
      humidityPct: 40,
      mq135Raw: 200,
      apiAQI: null, // simulates no internet — forces wearable_proxy fallback
      apiWeather: null,
      timestamp,
    },
  };

  return presets[scenario];
}