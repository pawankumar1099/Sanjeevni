// ============================================================================
// ai-engine/environment/heatIndex.ts
//
// Ambient temperature + relative humidity -> apparent heat index.
//
// Method:
//   Rothfusz regression / National Weather Service heat-index method.
//
// IMPORTANT:
//   This is an environmental heat-stress indicator.
//   It is NOT a measurement of core body temperature or dehydration.
//
// Pure TypeScript.
// No React Native / Node dependencies.
// ============================================================================

const MIN_TEMP_C = -50;
const MAX_TEMP_C = 60;

const MIN_HUMIDITY_PCT = 0;
const MAX_HUMIDITY_PCT = 100;

const ROTHFUSZ_MIN_F = 80;

/**
 * Convert Celsius to Fahrenheit.
 *
 * WHY:
 * The canonical Rothfusz equation is defined in Fahrenheit.
 */
function celsiusToFahrenheit(tempC: number): number {
  return (tempC * 9) / 5 + 32;
}

/**
 * Convert Fahrenheit to Celsius.
 */
function fahrenheitToCelsius(tempF: number): number {
  return ((tempF - 32) * 5) / 9;
}

function isValidTemperature(tempC: number): boolean {
  return (
    Number.isFinite(tempC) &&
    tempC >= MIN_TEMP_C &&
    tempC <= MAX_TEMP_C
  );
}

function isValidHumidity(humidityPct: number): boolean {
  return (
    Number.isFinite(humidityPct) &&
    humidityPct >= MIN_HUMIDITY_PCT &&
    humidityPct <= MAX_HUMIDITY_PCT
  );
}

/**
 * Calculate heat index from ambient temperature and humidity.
 *
 * Returns null when the environmental input is unavailable or invalid.
 *
 * WHY:
 * Returning null instead of 0 prevents missing environmental data from
 * being interpreted downstream as a perfectly safe temperature.
 */
export function calculateHeatIndexC(
  tempC: number | null,
  humidityPct: number | null,
): number | null {
  if (
    tempC === null ||
    humidityPct === null ||
    !isValidTemperature(tempC) ||
    !isValidHumidity(humidityPct)
  ) {
    return null;
  }

  const tempF = celsiusToFahrenheit(tempC);
  const rh = humidityPct;

  /**
   * The NWS method uses a simpler approximation below approximately
   * 80°F because the full Rothfusz regression is not intended for
   * that lower-temperature region.
   */
  const simpleHeatIndexF =
    0.5 *
    (
      tempF +
      61.0 +
      (tempF - 68.0) * 1.2 +
      rh * 0.094
    );

  /**
   * The operational NWS procedure averages the simple heat-index
   * estimate with the actual temperature before deciding whether
   * the full regression should be applied.
   */
  const averagedSimpleHI =
    (simpleHeatIndexF + tempF) / 2;

  if (averagedSimpleHI < ROTHFUSZ_MIN_F) {
    return fahrenheitToCelsius(averagedSimpleHI);
  }

  /**
   * Rothfusz regression.
   *
   * T  = Fahrenheit temperature
   * RH = relative humidity percentage
   */
  let heatIndexF =
    -42.379 +
    2.04901523 * tempF +
    10.14333127 * rh -
    0.22475541 * tempF * rh -
    0.00683783 * tempF * tempF -
    0.05481717 * rh * rh +
    0.00122874 * tempF * tempF * rh +
    0.00085282 * tempF * rh * rh -
    0.00000199 * tempF * tempF * rh * rh;

  /**
   * Low-humidity adjustment from the NWS formulation.
   *
   * WHY:
   * The regression needs a correction in this particular
   * temperature/humidity region.
   */
  if (
    rh < 13 &&
    tempF >= 80 &&
    tempF <= 112
  ) {
    const adjustment =
      ((13 - rh) / 4) *
      Math.sqrt(
        Math.max(
          0,
          (17 - Math.abs(tempF - 95)) / 17,
        ),
      );

    heatIndexF -= adjustment;
  }

  /**
   * High-humidity adjustment from the NWS formulation.
   */
  if (
    rh > 85 &&
    tempF >= 80 &&
    tempF <= 87
  ) {
    const adjustment =
      ((rh - 85) / 10) *
      ((87 - tempF) / 5);

    heatIndexF += adjustment;
  }

  const heatIndexC =
    fahrenheitToCelsius(heatIndexF);

  return Number.isFinite(heatIndexC)
    ? heatIndexC
    : null;
}