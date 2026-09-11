/**
 * Motion activity classification.
 *
 * WHY:
 * Motion state is important because downstream risk engines must know
 * whether an abnormal physiological signal happened during:
 *
 * - rest
 * - normal activity
 * - vigorous activity
 * - a possible fall
 *
 * This module intentionally does NOT diagnose anything.
 */

import type { MotionSample } from "../types";

export type ActivityState =
  | "REST"
  | "LIGHT_ACTIVITY"
  | "MODERATE_ACTIVITY"
  | "VIGOROUS_ACTIVITY"
  | "UNKNOWN";

export interface ActivityStateOutput {
  state: ActivityState;

  /**
   * Overall confidence in the activity classification.
   * Range: 0..1
   */
  confidence: number;

  /**
   * Mean acceleration magnitude in g.
   * null when the input cannot be evaluated.
   */
  accelerationMagnitudeG: number | null;

  /**
   * Dynamic acceleration RMS in g.
   * This removes the approximate gravity component and is
   * more useful for activity intensity.
   */
  dynamicAccelerationRmsG: number | null;

  /**
   * Gyroscope RMS in degrees/sec.
   *
   * null when gyro data is unavailable/invalid.
   */
  gyroRmsDps: number | null;

  /**
   * Number of valid motion samples used.
   */
  validSamples: number;
}

const GRAVITY_G = 1;

/**
 * Defensive bounds.
 *
 * WHY:
 * Extremely large values are usually caused by packet corruption,
 * disconnected sensors, unit mismatch, or malformed data.
 */
const MAX_ACCELERATION_G = 16;
const MAX_GYRO_DPS = 4000;

/**
 * Minimum number of valid samples required for classification.
 *
 * WHY:
 * We do not classify activity from one or two samples because
 * transient noise can otherwise cause unnecessary state changes.
 */
const MIN_VALID_SAMPLES = 5;

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function accelerationMagnitudeG(
  x: number,
  y: number,
  z: number,
): number {
  return Math.sqrt(x * x + y * y + z * z);
}

function calculateMean(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }

  let sum = 0;

  for (const value of values) {
    sum += value;
  }

  const mean = sum / values.length;

  return Number.isFinite(mean) ? mean : null;
}

function calculateRms(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }

  let sumSquares = 0;

  for (const value of values) {
    sumSquares += value * value;
  }

  const rms = Math.sqrt(sumSquares / values.length);

  return Number.isFinite(rms) ? rms : null;
}

export function classifyActivity(
  dynamicRmsG: number,
  gyroRmsDps: number | null,
): ActivityState {
  /**
   * The acceleration thresholds are intentionally broad.
   *
   * WHY:
   * This is an activity classifier, not a clinical-grade motion
   * measurement system. Individual placement and garment fit can
   * significantly change raw acceleration.
   */

  if (dynamicRmsG < 0.03) {
    return "REST";
  }

  if (dynamicRmsG < 0.10) {
    return "LIGHT_ACTIVITY";
  }

  if (dynamicRmsG < 0.25) {
    return "MODERATE_ACTIVITY";
  }

  /**
   * Gyroscope information is used as supporting evidence.
   *
   * WHY:
   * A large acceleration RMS with almost no rotational movement
   * can sometimes be caused by sensor placement or vibration.
   */
  if (
    gyroRmsDps !== null &&
    gyroRmsDps < 15 &&
    dynamicRmsG < 0.35
  ) {
    return "MODERATE_ACTIVITY";
  }

  return "VIGOROUS_ACTIVITY";
}

/**
 * Analyze one motion window.
 *
 * This function is stateless and therefore safe to call directly
 * from processSensorTick().
 */
export function analyzeActivity(
  input: MotionSample,
): ActivityStateOutput {
  if (
    !input ||
    !Array.isArray(input.accel) ||
    !Number.isFinite(input.sampleRate) ||
    input.sampleRate <= 0
  ) {
    return {
      state: "UNKNOWN",
      confidence: 0,
      accelerationMagnitudeG: null,
      dynamicAccelerationRmsG: null,
      gyroRmsDps: null,
      validSamples: 0,
    };
  }

  const accelerationMagnitudes: number[] = [];
  const dynamicAcceleration: number[] = [];
  const gyroMagnitudes: number[] = [];

  /**
   * We only process the paired portion of accel/gyro data.
   *
   * WHY:
   * The MotionSample interface does not explicitly guarantee that
   * accel and gyro arrays have identical lengths.
   */
  const pairedLength = Math.min(
    input.accel.length,
    input.gyro.length,
  );

  let validAccelSamples = 0;

  for (let i = 0; i < input.accel.length; i += 1) {
    const sample = input.accel[i];

    if (
      !sample ||
      !isFiniteNumber(sample.x) ||
      !isFiniteNumber(sample.y) ||
      !isFiniteNumber(sample.z)
    ) {
      continue;
    }

    const magnitude = accelerationMagnitudeG(
      sample.x,
      sample.y,
      sample.z,
    );

    if (
      !Number.isFinite(magnitude) ||
      magnitude > MAX_ACCELERATION_G
    ) {
      continue;
    }

    accelerationMagnitudes.push(magnitude);

    /**
     * Dynamic acceleration:
     *
     * |a| - 1g
     *
     * This is a simple activity-oriented approximation.
     *
     * WHY:
     * The accelerometer measures both gravity and body movement.
     * Removing approximately 1g prevents a stationary person from
     * being interpreted as continuously active.
     */
    const dynamic = Math.abs(magnitude - GRAVITY_G);

    if (Number.isFinite(dynamic)) {
      dynamicAcceleration.push(dynamic);
    }

    validAccelSamples += 1;
  }

  for (let i = 0; i < pairedLength; i += 1) {
    const gyro = input.gyro[i];

    if (
      !gyro ||
      !isFiniteNumber(gyro.x) ||
      !isFiniteNumber(gyro.y) ||
      !isFiniteNumber(gyro.z)
    ) {
      continue;
    }

    const magnitude = Math.sqrt(
      gyro.x * gyro.x +
        gyro.y * gyro.y +
        gyro.z * gyro.z,
    );

    if (
      !Number.isFinite(magnitude) ||
      magnitude > MAX_GYRO_DPS
    ) {
      continue;
    }

    gyroMagnitudes.push(magnitude);
  }

  if (validAccelSamples < MIN_VALID_SAMPLES) {
    return {
      state: "UNKNOWN",
      confidence: 0,
      accelerationMagnitudeG: null,
      dynamicAccelerationRmsG: null,
      gyroRmsDps:
        gyroMagnitudes.length > 0
          ? calculateRms(gyroMagnitudes)
          : null,
      validSamples: validAccelSamples,
    };
  }

  const accelerationMagnitude = calculateMean(
    accelerationMagnitudes,
  );

  const dynamicRms = calculateRms(dynamicAcceleration);

  const gyroRms =
    gyroMagnitudes.length > 0
      ? calculateRms(gyroMagnitudes)
      : null;

  if (dynamicRms === null) {
    return {
      state: "UNKNOWN",
      confidence: 0,
      accelerationMagnitudeG: accelerationMagnitude,
      dynamicAccelerationRmsG: null,
      gyroRmsDps: gyroRms,
      validSamples: validAccelSamples,
    };
  }

  const state = classifyActivity(dynamicRms, gyroRms);

  /**
   * Confidence increases with:
   * - more valid samples
   * - availability of gyro
   *
   * WHY:
   * A classifier based only on a tiny window should not have the
   * same confidence as one supported by a complete sensor window.
   */
  const sampleConfidence = clamp01(
    validAccelSamples / Math.max(input.accel.length, 1),
  );

  const gyroConfidence =
    input.gyro.length > 0
      ? clamp01(gyroMagnitudes.length / input.gyro.length)
      : 0;

  const confidence =
    gyroMagnitudes.length > 0
      ? 0.75 * sampleConfidence + 0.25 * gyroConfidence
      : 0.85 * sampleConfidence;

  return {
    state,
    confidence: clamp01(confidence),
    accelerationMagnitudeG: accelerationMagnitude,
    dynamicAccelerationRmsG: dynamicRms,
    gyroRmsDps: gyroRms,
    validSamples: validAccelSamples,
  };
}