import type { EnvironmentSample } from "../types";
import { clamp } from "../utils/math";

/**
 * AQI data source.
 *
 * "api"            → externally supplied AQI value
 * "wearable_proxy" → uncalibrated MQ135-derived proxy
 * "unavailable"    → no usable AQI data
 */
export type DataSource =
  | "api"
  | "wearable_proxy"
  | "unavailable";

export interface AQIResolution {
  /**
   * Resolved AQI value.
   *
   * null means that no usable AQI value
   * could be obtained.
   */
  aqi: number | null;

  /** Source of the AQI value. */
  source: DataSource;

  /**
   * Confidence in the AQI source.
   *
   * 0.0 → unusable
   * 1.0 → highest confidence
   */
  confidence: number;
}

// CPCB AQI operational range.
const MIN_AQI = 0;
const MAX_AQI = 500;

// API AQI is preferred because it is already an AQI value.
const API_CONFIDENCE = 0.95;

// MQ135 raw value is NOT a calibrated CPCB AQI.
const MQ135_PROXY_CONFIDENCE = 0.20;

/**
 * Check whether a value is a valid finite number.
 *
 * Kept local deliberately so this file does not require
 * changes to the existing utils/math.ts API.
 */
function isFiniteNumber(
  value: number | null | undefined,
): value is number {
  return (
    value !== null &&
    value !== undefined &&
    Number.isFinite(value)
  );
}

/**
 * Convert a raw MQ135 ADC reading into a low-confidence
 * AQI-like proxy.
 *
 * IMPORTANT:
 * This is NOT a scientifically valid CPCB AQI calculation.
 *
 * A true AQI calculation requires pollutant-specific
 * concentration measurements and the applicable CPCB
 * breakpoint tables.
 *
 * This proxy is only used as a fallback environmental signal.
 */
function normalizeMQ135Proxy(
  rawValue: number,
): number | null {
  if (!Number.isFinite(rawValue)) {
    return null;
  }

  /**
   * Current hardware contract uses a 0–1023-style raw value.
   *
   * If the ESP32 implementation later changes to a 12-bit
   * 0–4095 ADC value, this conversion must be updated
   * together with the hardware protocol.
   */
  if (rawValue < 0 || rawValue > 1023) {
    return null;
  }

  const proxyAQI =
    (rawValue / 1023) * MAX_AQI;

  return clamp(
    proxyAQI,
    MIN_AQI,
    MAX_AQI,
  );
}

/**
 * Resolve the best available AQI source.
 *
 * Priority:
 * 1. API AQI
 * 2. MQ135 low-confidence proxy
 * 3. unavailable
 */
export function resolveAQI(
  environment: EnvironmentSample,
): AQIResolution {
  /**
   * Environment data itself should normally always exist,
   * but this defensive check prevents runtime failures.
   */
  if (!environment) {
    return {
      aqi: null,
      source: "unavailable",
      confidence: 0,
    };
  }

  /**
   * API AQI.
   *
   * IMPORTANT:
   * We explicitly narrow apiAQI before returning it.
   *
   * This prevents:
   *
   * "'environment.apiAQI' is possibly 'null'"
   */
  const apiAQI = environment.apiAQI;

  if (
    isFiniteNumber(apiAQI) &&
    apiAQI >= MIN_AQI &&
    apiAQI <= MAX_AQI
  ) {
    return {
      aqi: apiAQI,
      source: "api",
      confidence: API_CONFIDENCE,
    };
  }

  /**
   * MQ135 fallback.
   *
   * mq135Raw is number | null, so explicitly check
   * for null before passing it to the proxy function.
   */
  const mq135Raw = environment.mq135Raw;

  if (isFiniteNumber(mq135Raw)) {
    const wearableProxy =
      normalizeMQ135Proxy(mq135Raw);

    if (wearableProxy !== null) {
      return {
        aqi: wearableProxy,
        source: "wearable_proxy",
        confidence: MQ135_PROXY_CONFIDENCE,
      };
    }
  }

  /**
   * Never return 0 when data is unavailable.
   *
   * 0 is a legitimate AQI and would incorrectly imply
   * excellent air quality.
   */
  return {
    aqi: null,
    source: "unavailable",
    confidence: 0,
  };
}