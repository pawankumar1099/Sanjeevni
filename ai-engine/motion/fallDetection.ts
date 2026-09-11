/**
 * Fall detection using accelerometer + gyroscope signals.
 *
 * IMPORTANT:
 * This module detects a FALL-LIKE MOTION EVENT.
 * It does not diagnose injury or medical condition.
 *
 * The detector uses:
 *
 * 1. acceleration impact evidence
 * 2. rotational-motion evidence when available
 * 3. post-event inactivity evidence
 * 4. temporal persistence
 *
 * WHY:
 * A single acceleration spike can come from:
 * - jumping
 * - sitting down quickly
 * - dropping the sensor
 * - hitting furniture
 * - packet corruption
 *
 * Therefore acceleration alone must never trigger a final alert.
 */

import type { MotionSample } from "../types";

export type FallDetectionState =
  | "NO_EVENT"
  | "POSSIBLE_IMPACT"
  | "RECOVERY_CHECK"
  | "FALL_CANDIDATE"
  | "UNKNOWN";

export interface FallDetectionOutput {
  detected: boolean;

  state: FallDetectionState;

  /**
   * Confidence range: 0..1.
   */
  confidence: number;

  /**
   * Peak acceleration magnitude in g.
   */
  peakAccelerationG: number | null;

  /**
   * Peak gyroscope magnitude in dps.
   */
  peakGyroDps: number | null;

  /**
   * Indicates whether post-impact inactivity was observed.
   */
  postImpactInactivity: boolean;

  /**
   * Number of valid accelerometer samples.
   */
  validSamples: number;
}

/**
 * Approximate fall-event thresholds.
 *
 * These are deliberately used as event evidence rather than
 * standalone medical thresholds.
 */
const IMPACT_THRESHOLD_G = 2.5;
const STRONG_IMPACT_THRESHOLD_G = 3.0;

/**
 * Rotation can provide supporting evidence that the body
 * orientation changed rapidly.
 */
const ROTATION_SUPPORT_THRESHOLD_DPS = 150;

/**
 * After an impact, a short low-motion period provides useful
 * evidence that the person may have fallen rather than simply
 * performed a vigorous movement.
 */
const POST_IMPACT_WINDOW_MS = 1500;

/**
 * Motion below this dynamic acceleration is considered relatively
 * inactive for the fall-recovery check.
 */
const INACTIVITY_DYNAMIC_G = 0.08;

/**
 * At least this many samples must be available before the module
 * can make a fall decision.
 */
const MIN_VALID_SAMPLES = 10;

const MAX_ACCELERATION_G = 16;
const MAX_GYRO_DPS = 4000;

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function magnitude3D(
  x: number,
  y: number,
  z: number,
): number {
  return Math.sqrt(x * x + y * y + z * z);
}

function dynamicAccelerationG(
  x: number,
  y: number,
  z: number,
): number {
  return Math.abs(magnitude3D(x, y, z) - 1);
}

/**
 * Detect a fall-like event in one motion window.
 *
 * NOTE:
 * The function is stateless. Temporal persistence across multiple
 * calls should be handled by the decision layer / falseAlarmGate.
 */
export function detectFall(
  input: MotionSample,
): FallDetectionOutput {
  if (
    !input ||
    !Array.isArray(input.accel) ||
    !Number.isFinite(input.sampleRate) ||
    input.sampleRate <= 0
  ) {
    return {
      detected: false,
      state: "UNKNOWN",
      confidence: 0,
      peakAccelerationG: null,
      peakGyroDps: null,
      postImpactInactivity: false,
      validSamples: 0,
    };
  }

  const samplePeriodMs = 1000 / input.sampleRate;

  const validAccel: {
    index: number;
    magnitudeG: number;
    dynamicG: number;
  }[] = [];

  let peakAccelerationG = 0;

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

    const magnitudeG = magnitude3D(
      sample.x,
      sample.y,
      sample.z,
    );

    if (
      !Number.isFinite(magnitudeG) ||
      magnitudeG > MAX_ACCELERATION_G
    ) {
      continue;
    }

    const dynamicG = dynamicAccelerationG(
      sample.x,
      sample.y,
      sample.z,
    );

    validAccel.push({
      index: i,
      magnitudeG,
      dynamicG,
    });

    peakAccelerationG = Math.max(
      peakAccelerationG,
      magnitudeG,
    );
  }

  if (validAccel.length < MIN_VALID_SAMPLES) {
    return {
      detected: false,
      state: "UNKNOWN",
      confidence: 0,
      peakAccelerationG:
        validAccel.length > 0 ? peakAccelerationG : null,
      peakGyroDps: null,
      postImpactInactivity: false,
      validSamples: validAccel.length,
    };
  }

  /**
   * Find the strongest acceleration event.
   */
  let impactPosition = -1;

  for (let i = 0; i < validAccel.length; i += 1) {
    if (
      validAccel[i].magnitudeG >= IMPACT_THRESHOLD_G
    ) {
      impactPosition = i;
      break;
    }
  }

  /**
   * No impact evidence.
   *
   * WHY:
   * We intentionally do not label vigorous activity as a fall.
   */
  if (impactPosition === -1) {
    return {
      detected: false,
      state: "NO_EVENT",
      confidence: 0,
      peakAccelerationG,
      peakGyroDps: calculatePeakGyro(input),
      postImpactInactivity: false,
      validSamples: validAccel.length,
    };
  }

  /**
   * Look for a post-impact low-motion period.
   */
  const maxRecoverySamples = Math.max(
    1,
    Math.ceil(
      POST_IMPACT_WINDOW_MS / samplePeriodMs,
    ),
  );

  let recoverySamples = 0;
  let inactivitySamples = 0;

  for (
    let i = impactPosition + 1;
    i < validAccel.length &&
    i <= impactPosition + maxRecoverySamples;
    i += 1
  ) {
    recoverySamples += 1;

    if (
      validAccel[i].dynamicG <=
      INACTIVITY_DYNAMIC_G
    ) {
      inactivitySamples += 1;
    }
  }

  /**
   * Require enough post-impact evidence.
   */
  const recoveryCoverage =
    recoverySamples > 0
      ? inactivitySamples / recoverySamples
      : 0;

  const postImpactInactivity =
    recoverySamples >= 5 &&
    recoveryCoverage >= 0.60;

  const peakGyroDps = calculatePeakGyro(input);

  /**
   * Rotational movement is supporting evidence only.
   */
  const rotationEvidence =
    peakGyroDps !== null &&
    peakGyroDps >= ROTATION_SUPPORT_THRESHOLD_DPS;

  /**
   * Strong impact is stronger evidence than a barely-threshold
   * acceleration spike.
   */
  const strongImpact =
    peakAccelerationG >= STRONG_IMPACT_THRESHOLD_G;

  /**
   * Scoring is evidence aggregation, not a medical probability.
   *
   * WHY:
   * We want a transparent mechanism that can later be replaced
   * internally by a trained classifier without changing the
   * external module interface.
   */
  let score = 0;

  if (strongImpact) {
    score += 0.35;
  } else {
    score += 0.20;
  }

  if (rotationEvidence) {
    score += 0.20;
  }

  if (postImpactInactivity) {
    score += 0.40;
  }

  /**
   * A fall candidate requires post-impact inactivity.
   *
   * WHY:
   * Without it, events such as running, jumping or dropping the
   * sensor can resemble an impact.
   */
  const detected =
    postImpactInactivity &&
    score >= 0.60;

  const state: FallDetectionState = detected
    ? "FALL_CANDIDATE"
    : "RECOVERY_CHECK";

  /**
   * Confidence is based on evidence completeness.
   */
  const evidenceConfidence = clamp01(
    score +
      (validAccel.length >= input.accel.length
        ? 0.05
        : 0),
  );

  return {
    detected,
    state,
    confidence: detected
      ? evidenceConfidence
      : Math.min(evidenceConfidence, 0.59),
    peakAccelerationG,
    peakGyroDps,
    postImpactInactivity,
    validSamples: validAccel.length,
  };
}

function calculatePeakGyro(
  input: MotionSample,
): number | null {
  if (!Array.isArray(input.gyro)) {
    return null;
  }

  let peak = 0;
  let valid = 0;

  for (const sample of input.gyro) {
    if (
      !sample ||
      !isFiniteNumber(sample.x) ||
      !isFiniteNumber(sample.y) ||
      !isFiniteNumber(sample.z)
    ) {
      continue;
    }

    const magnitude = magnitude3D(
      sample.x,
      sample.y,
      sample.z,
    );

    if (
      !Number.isFinite(magnitude) ||
      magnitude > MAX_GYRO_DPS
    ) {
      continue;
    }

    peak = Math.max(peak, magnitude);
    valid += 1;
  }

  return valid > 0 ? peak : null;
}