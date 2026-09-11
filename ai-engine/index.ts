// ai-engine/index.ts
//
// Sanjeevni Phase-1 on-device AI orchestration layer.
//
// Architecture:
//
//   SensorTickInput
//        │
//        ├── Motion
//        │     ├── Activity classification
//        │     └── Fall-like event detection
//        │
//        ├── ECG
//        │     ├── Filtering
//        │     ├── SQI
//        │     ├── R-peak detection
//        │     ├── RR validation
//        │     ├── HR
//        │     ├── HRV
//        │     └── Rhythm anomaly
//        │
//        ├── PPG (optional / currently unavailable)
//        │     ├── Quality gate
//        │     └── SpO2
//        │
//        ├── Environment
//        │     ├── Heat index
//        │     ├── AQI resolution
//        │     └── Respiratory environmental risk
//        │
//        ▼
//   Personal Baseline
//        │
//        ▼
//   Sensor Fusion
//        │
//        ▼
//   False-Alarm Gate
//        │
//        ▼
//   Risk Decision Engine
//        │
//        ▼
//   SanjeevniRiskOutput
//
// HARD CONSTRAINTS:
// - Phase 1 only.
// - DSP + statistics + deterministic rules.
// - No ML/DL.
// - No React Native imports.
// - Missing values remain null internally.
// - Risk decisions are non-diagnostic.
// - Stateful components are isolated PER USER.
// ============================================================================

import type {
  ECGPipelineOutput,
  EnvironmentPipelineOutput,
  FalseAlarmGateInput,
  FusionInput,
  FusionOutput,
  MotionPipelineOutput,
  PPGPipelineOutput,
  RiskLevel,
  SanjeevniRiskOutput,
  SensorTickInput,
} from "./types";

// ============================================================================
// ECG
// ============================================================================

import { ECGFilterChain } from "./ecg/filters";
import { computeSQI } from "./ecg/sqi";
import { detectRPeaks } from "./ecg/rPeakDetection";
import { validateRRIntervals } from "./ecg/rrValidation";
import { computeHeartRate } from "./ecg/heartRate";
import { computeHRV } from "./ecg/hrv";
import { evaluateRhythmAnomaly } from "./ecg/rhythmAnomaly";

// ============================================================================
// PPG
// ============================================================================

import { estimateSpO2 } from "./ppg/spo2Estimator";

// ============================================================================
// MOTION
// ============================================================================

import { analyzeActivity } from "./motion/activityState";
import { detectFall } from "./motion/fallDetection";

// ============================================================================
// ENVIRONMENT
// ============================================================================

import { calculateHeatIndexC } from "./environment/heatIndex";
import { calculateRespiratoryRisk } from "./environment/respiratoryRisk";
import { resolveAQI } from "./environment/aqiSource";

// ============================================================================
// BASELINE
// ============================================================================

import { PersonalBaselineEngine } from "./baseline/personalBaseline";

// ============================================================================
// FUSION
// ============================================================================

import { fuseSensorRisks } from "./fusion/sensorFusion";

// ============================================================================
// DECISION
// ============================================================================

import { createFalseAlarmGate } from "./decision/falseAlarmGate";
import { createRiskDecisionEngine } from "./decision/riskDecisionEngine";

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Actual Sanjeevni hardware ECG rate.
 *
 * ESP32/BioAmp:
 * 500 Hz, 128 samples per packet.
 */
const ECG_SAMPLE_RATE_MIN_HZ = 100;
const ECG_SAMPLE_RATE_MAX_HZ = 1000;

/**
 * Maximum retained RR history.
 *
 * WHY:
 * Prevents unbounded memory growth while retaining enough recent
 * intervals for HR/HRV calculations.
 */
const MAX_RR_HISTORY = 120;

/**
 * Minimum RR intervals required before we treat a window as useful
 * for rhythm-anomaly persistence.
 */
const MIN_RR_FOR_RHYTHM_ANALYSIS = 3;

/**
 * Heat-index risk thresholds based on the standard NWS heat-index
 * interpretation bands.
 *
 * These are converted into a normalized 0..1 risk candidate.
 *
 * IMPORTANT:
 * This is environmental heat-stress context, NOT core body temperature.
 */
const HEAT_INDEX_CAUTION_C = 26.7; // 80°F
const HEAT_INDEX_EXTREME_CAUTION_C = 32.2; // 90°F
const HEAT_INDEX_DANGER_C = 40.6; // 105°F
const HEAT_INDEX_EXTREME_DANGER_C = 51.7; // 125°F

// ============================================================================
// PER-USER STATE
// ============================================================================
//
// IMPORTANT:
//
// ECGFilterChain, PersonalBaselineEngine and FalseAlarmGate are stateful.
//
// They MUST NOT be global shared instances.
//
// Otherwise:
//   User A -> User B
// could contaminate:
//   - ECG filter memory
//   - RR history
//   - personal baseline
//   - persistence counters
//   - cooldown state
//
// Therefore every user receives an isolated runtime.
// ============================================================================

interface RuntimeState {
  ecgFilter: ECGFilterChain;

  baseline: PersonalBaselineEngine;

  falseAlarmGate: ReturnType<typeof createFalseAlarmGate>;

  riskDecisionEngine: ReturnType<
    typeof createRiskDecisionEngine
  >;

  rrHistoryMs: number[];

  /**
   * Number of consecutive anomalous RR windows.
   *
   * rhythmAnomaly.ts is intentionally stateless, so persistence
   * belongs here.
   */
  consecutiveAnomalousWindows: number;

  /**
   * Last accepted top-level timestamp.
   */
  lastTimestamp: number | null;

  /**
   * Whether at least one valid chronological tick has been processed.
   */
  initialized: boolean;
}

const runtimeByUser = new Map<
  string,
  RuntimeState
>();

function createRuntime(): RuntimeState {
  return {
    ecgFilter: new ECGFilterChain(),

    baseline:
      new PersonalBaselineEngine(),

    falseAlarmGate:
      createFalseAlarmGate(),

    riskDecisionEngine:
      createRiskDecisionEngine(),

    rrHistoryMs: [],

    consecutiveAnomalousWindows: 0,

    lastTimestamp: null,

    initialized: false,
  };
}

function getRuntime(
  userId: string,
): RuntimeState {
  const existing =
    runtimeByUser.get(userId);

  if (existing) {
    return existing;
  }

  const runtime =
    createRuntime();

  runtimeByUser.set(
    userId,
    runtime,
  );

  return runtime;
}

// ============================================================================
// NUMERIC HELPERS
// ============================================================================

function isFiniteNumber(
  value: unknown,
): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  );
}

function clamp01(
  value: number,
): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(
    0,
    Math.min(1, value),
  );
}

/**
 * Safe mean.
 *
 * Returns null rather than 0 because 0 would look like real data.
 */
function safeMean(
  values: number[],
): number | null {
  if (values.length === 0) {
    return null;
  }

  let sum = 0;

  for (const value of values) {
    if (!Number.isFinite(value)) {
      return null;
    }

    sum += value;
  }

  const result =
    sum / values.length;

  return Number.isFinite(result)
    ? result
    : null;
}

// ============================================================================
// EMPTY OUTPUTS
// ============================================================================
//
// These are deliberately conservative.
//
// Missing sensor data is NOT represented as a healthy measurement.
// ============================================================================

function emptyECGOutput(): ECGPipelineOutput {
  return {
    heartRate: null,

    rrIntervals: [],

    hrv: null,

    rhythmFlag: null,

    sqi: {
      label: "INVALID",
      score: 0,
    },
  };
}

function emptyPPGOutput(): PPGPipelineOutput {
  return {
    spo2: null,

    spo2Confidence: 0,

    ppgQuality: "UNAVAILABLE",

    ppgDerivedHR: null,
  };
}

function emptyMotionOutput(): MotionPipelineOutput {
  return {
    motionLevel: 0,

    /**
     * The public types.ts contract does not have UNKNOWN.
     *
     * Therefore the adapter uses REST as the safe public fallback.
     * The internal activity module still returns UNKNOWN when appropriate.
     */
    activityState: "REST",

    fall: {
      detected: false,
      confidence: 0,
      stage: "NONE",
    },
  };
}

function emptyEnvironmentOutput(): EnvironmentPipelineOutput {
  return {
    heatIndexC: null,

    heatIndexSource: "unavailable",

    aqi: null,

    aqiSource: "unavailable",

    aqiConfidence: 0,
  };
}

// ============================================================================
// MOTION ADAPTERS
// ============================================================================

/**
 * Convert the richer activityState.ts vocabulary into the locked
 * types.ts vocabulary.
 *
 * activityState.ts:
 *
 *   REST
 *   LIGHT_ACTIVITY
 *   MODERATE_ACTIVITY
 *   VIGOROUS_ACTIVITY
 *   UNKNOWN
 *
 * types.ts:
 *
 *   REST
 *   LIGHT
 *   ACTIVE
 *
 * WHY:
 * The internal motion classifier can remain more expressive without
 * changing the public engine contract.
 */
function mapActivityState(
  state:
    | "REST"
    | "LIGHT_ACTIVITY"
    | "MODERATE_ACTIVITY"
    | "VIGOROUS_ACTIVITY"
    | "UNKNOWN",
):
  | "REST"
  | "LIGHT"
  | "ACTIVE" {
  switch (state) {
    case "REST":
      return "REST";

    case "LIGHT_ACTIVITY":
      return "LIGHT";

    case "MODERATE_ACTIVITY":
    case "VIGOROUS_ACTIVITY":
      return "ACTIVE";

    case "UNKNOWN":
    default:
      return "REST";
  }
}

/**
 * Convert dynamic acceleration RMS into the normalized motion level
 * expected by SQI and the public types.ts contract.
 *
 * 0.35g is the upper activity range used by activityState.ts before
 * vigorous activity dominates.
 *
 * This is a context/intensity normalization, not a clinical metric.
 */
function calculateMotionLevel(
  dynamicAccelerationRmsG: number | null,
): number {
  if (
    dynamicAccelerationRmsG === null ||
    !Number.isFinite(
      dynamicAccelerationRmsG,
    )
  ) {
    return 0;
  }

  return clamp01(
    dynamicAccelerationRmsG / 0.35,
  );
}

/**
 * Adapt fallDetection.ts states into the locked types.ts FallStage.
 *
 * Internal detector:
 *
 *   NO_EVENT
 *   POSSIBLE_IMPACT
 *   RECOVERY_CHECK
 *   FALL_CANDIDATE
 *   UNKNOWN
 *
 * Public contract:
 *
 *   NONE
 *   FREEFALL
 *   IMPACT
 *   CONFIRMED
 */
function mapFallStage(
  state:
    | "NO_EVENT"
    | "POSSIBLE_IMPACT"
    | "RECOVERY_CHECK"
    | "FALL_CANDIDATE"
    | "UNKNOWN",
):
  | "NONE"
  | "FREEFALL"
  | "IMPACT"
  | "CONFIRMED" {
  switch (state) {
    case "POSSIBLE_IMPACT":
      return "IMPACT";

    case "RECOVERY_CHECK":
      return "IMPACT";

    case "FALL_CANDIDATE":
      return "CONFIRMED";

    case "NO_EVENT":
    case "UNKNOWN":
    default:
      return "NONE";
  }
}

// ============================================================================
// MOTION PIPELINE
// ============================================================================

function processMotion(
  input: SensorTickInput,
): MotionPipelineOutput {
  if (!input.motion) {
    return emptyMotionOutput();
  }

  if (
    !Array.isArray(
      input.motion.accel,
    ) ||
    input.motion.accel.length === 0
  ) {
    return emptyMotionOutput();
  }

  /**
   * IMPORTANT:
   *
   * analyzeActivity() is the correct API.
   *
   * classifyActivity() returns only ActivityState.
   *
   * analyzeActivity() returns:
   *   state
   *   confidence
   *   accelerationMagnitudeG
   *   dynamicAccelerationRmsG
   *   gyroRmsDps
   *   validSamples
   */
  const activity =
    analyzeActivity(
      input.motion,
    );

  const fall =
    detectFall(
      input.motion,
    );

  const activityState =
    mapActivityState(
      activity.state,
    );

  const motionLevel =
    calculateMotionLevel(
      activity.dynamicAccelerationRmsG,
    );

  return {
    motionLevel,

    activityState,

    fall: {
      detected:
        Boolean(fall.detected),

      confidence:
        clamp01(
          fall.confidence,
        ),

      stage:
        mapFallStage(
          fall.state,
        ),
    },
  };
}

// ============================================================================
// ECG PIPELINE
// ============================================================================

function processECG(
  input: SensorTickInput,
  runtime: RuntimeState,
  motion: MotionPipelineOutput,
): ECGPipelineOutput {
  if (!input.ecg) {
    runtime.consecutiveAnomalousWindows = 0;

    return emptyECGOutput();
  }

  const ecg =
    input.ecg;

  // --------------------------------------------------------------------------
  // Basic validation
  // --------------------------------------------------------------------------

  if (
    !Array.isArray(
      ecg.samples,
    ) ||
    ecg.samples.length < 10 ||
    !isFiniteNumber(
      ecg.sampleRate,
    ) ||
    ecg.sampleRate <= 0
  ) {
    runtime.consecutiveAnomalousWindows = 0;

    return emptyECGOutput();
  }

  /**
   * Prevent malformed timing information from entering the ECG pipeline.
   */
  if (
    ecg.sampleRate <
      ECG_SAMPLE_RATE_MIN_HZ ||
    ecg.sampleRate >
      ECG_SAMPLE_RATE_MAX_HZ
  ) {
    runtime.consecutiveAnomalousWindows = 0;

    return emptyECGOutput();
  }

  // --------------------------------------------------------------------------
  // 1. FILTER
  // --------------------------------------------------------------------------

  const filtered =
    runtime.ecgFilter.process(
      ecg,
    );

  // --------------------------------------------------------------------------
  // 2. SIGNAL QUALITY
  //
  // Motion is deliberately supplied to SQI.
  //
  // This fixes the previous bug where motionLevel was calculated but
  // never actually reached the ECG quality gate.
  // --------------------------------------------------------------------------

  const sqi =
    computeSQI(
      filtered,
      {
        motionLevel:
          motion.motionLevel,
      },
    );

  // --------------------------------------------------------------------------
  // 3. HARD QUALITY GATE
  // --------------------------------------------------------------------------

  if (
    sqi.label === "INVALID" ||
    sqi.label ===
      "MOTION_CORRUPTED"
  ) {
    runtime.consecutiveAnomalousWindows = 0;

    return {
      heartRate: null,

      rrIntervals: [],

      hrv: null,

      rhythmFlag: null,

      sqi,
    };
  }

  // --------------------------------------------------------------------------
  // 4. R-PEAK DETECTION
  // --------------------------------------------------------------------------

  const peaks =
    detectRPeaks(
      filtered,
      {
        timestampStart:
          ecg.timestampStart,
      },
    );

  // --------------------------------------------------------------------------
  // 5. RR VALIDATION
  // --------------------------------------------------------------------------

  const rrResult =
    validateRRIntervals(
      peaks,
    );

  if (
    rrResult.rrIntervalsMs.length ===
    0
  ) {
    runtime.consecutiveAnomalousWindows = 0;

    return {
      heartRate: null,

      rrIntervals: [],

      hrv: null,

      rhythmFlag: null,

      sqi,
    };
  }

  // --------------------------------------------------------------------------
  // 6. UPDATE SHORT-TERM RR HISTORY
  // --------------------------------------------------------------------------

  runtime.rrHistoryMs.push(
    ...rrResult.rrIntervalsMs,
  );

  if (
    runtime.rrHistoryMs.length >
    MAX_RR_HISTORY
  ) {
    runtime.rrHistoryMs =
      runtime.rrHistoryMs.slice(
        -MAX_RR_HISTORY,
      );
  }

  // --------------------------------------------------------------------------
  // 7. HEART RATE
  // --------------------------------------------------------------------------

  const heartRate =
    computeHeartRate(
      runtime.rrHistoryMs,
    );

  // --------------------------------------------------------------------------
  // 8. HRV
  // --------------------------------------------------------------------------

  const hrv =
    computeHRV(
      runtime.rrHistoryMs,
    );

  // --------------------------------------------------------------------------
  // 9. RHYTHM ANOMALY
  //
  // Baseline MUST be read BEFORE the current observation is added.
  //
  // Otherwise the current abnormal window could contaminate its own
  // comparison baseline.
  // --------------------------------------------------------------------------

  let rhythmFlag =
    null;

  const currentRRMean =
    safeMean(
      rrResult.rrIntervalsMs,
    );

  if (
    currentRRMean !== null &&
    rrResult.rrIntervalsMs.length >=
      MIN_RR_FOR_RHYTHM_ANALYSIS
  ) {
    const baselineSnapshot =
      runtime.baseline.getSnapshot();

    const rrBaseline =
      baselineSnapshot.rrInterval;

    /**
     * Only evaluate against an established personal RR baseline.
     */
    if (
      rrBaseline.ready &&
      rrBaseline.mean !== null &&
      rrBaseline.stdDev !== null
    ) {
      const rrZScore =
        rrBaseline.stdDev >
        0
          ? Math.abs(
              (
                currentRRMean -
                rrBaseline.mean
              ) /
                rrBaseline.stdDev,
            )
          : null;

      if (
        rrZScore !== null &&
        Number.isFinite(rrZScore) &&
        rrZScore >= 2.5
      ) {
        runtime.consecutiveAnomalousWindows +=
          1;
      } else {
        runtime.consecutiveAnomalousWindows = 0;
      }

      rhythmFlag =
        evaluateRhythmAnomaly(
          rrResult.rrIntervalsMs,
          {
            baselineMeanRR:
              rrBaseline.mean,

            baselineStdDevRR:
              rrBaseline.stdDev,

            hasSufficientBaselineHistory:
              rrBaseline.ready,

            consecutiveAnomalousWindows:
              runtime.consecutiveAnomalousWindows,
          },
        );
    } else {
      runtime.consecutiveAnomalousWindows = 0;

      /**
       * We intentionally do not call rhythm anomaly logic without
       * sufficient personal baseline history.
       */
      rhythmFlag = null;
    }
  } else {
    runtime.consecutiveAnomalousWindows = 0;
  }

  return {
    heartRate,

    rrIntervals:
      rrResult.rrIntervalsMs,

    hrv,

    rhythmFlag,

    sqi,
  };
}

// ============================================================================
// PPG PIPELINE
// ============================================================================

function processPPG(
  input: SensorTickInput,
  motion: MotionPipelineOutput,
): PPGPipelineOutput {
  /**
   * Current Sanjeevni hardware does not contain PPG.
   *
   * Therefore the normal Phase-1 output is:
   *
   *   spo2 = null
   *   confidence = 0
   *
   * If PPG is added later, this same orchestration path already supports it.
   */

  if (!input.ppg) {
    return emptyPPGOutput();
  }

  return estimateSpO2(
    input.ppg,
    {
      motionLevel:
        motion.motionLevel,
    },
  );
}

// ============================================================================
// ENVIRONMENT PIPELINE
// ============================================================================

function processEnvironment(
  input: SensorTickInput,
): EnvironmentPipelineOutput {
  if (!input.environment) {
    return emptyEnvironmentOutput();
  }

  const environment =
    input.environment;

  // --------------------------------------------------------------------------
  // AQI
  //
  // resolveAQI() handles:
  //   API AQI
  //   MQ135 wearable proxy
  //   unavailable
  //
  // MQ135 proxy confidence remains deliberately low.
  // --------------------------------------------------------------------------

  const aqiResolution =
    resolveAQI(
      environment,
    );

  // --------------------------------------------------------------------------
  // HEAT INDEX
  //
  // Prefer wearable temperature/humidity.
  //
  // If wearable environmental values are unavailable, use API weather
  // values when present.
  // --------------------------------------------------------------------------

  let heatTempC:
    number | null = null;

  let heatHumidityPct:
    number | null = null;

  let heatSource:
    "wearable" | "api" | "unavailable" =
      "unavailable";

  if (
    isFiniteNumber(
      environment.tempC,
    ) &&
    isFiniteNumber(
      environment.humidityPct,
    )
  ) {
    heatTempC =
      environment.tempC;

    heatHumidityPct =
      environment.humidityPct;

    heatSource =
      "wearable";
  } else if (
    environment.apiWeather &&
    isFiniteNumber(
      environment.apiWeather.tempC,
    ) &&
    isFiniteNumber(
      environment.apiWeather.humidityPct,
    )
  ) {
    heatTempC =
      environment.apiWeather.tempC;

    heatHumidityPct =
      environment.apiWeather.humidityPct;

    heatSource =
      "api";
  }

  const heatIndexC =
    calculateHeatIndexC(
      heatTempC,
      heatHumidityPct,
    );

  return {
    heatIndexC,

    heatIndexSource:
      heatIndexC === null
        ? "unavailable"
        : heatSource,

    aqi:
      aqiResolution.aqi,

    aqiSource:
      aqiResolution.source,

    aqiConfidence:
      clamp01(
        aqiResolution.confidence,
      ),
  };
}

// ============================================================================
// HEAT RISK TRANSFORMATION
// ============================================================================
//
// sensorFusion.ts expects a normalized risk score 0..1.
//
// It does NOT expect the raw heat-index temperature.
//
// Therefore this adapter converts:
//
//   heat index °C
//
// into:
//
//   normalized heat-risk candidate
//
// This is deliberately environmental risk context, not diagnosis.
// ============================================================================

function calculateHeatRiskScore(
  heatIndexC: number | null,
): number | null {
  if (
    heatIndexC === null ||
    !Number.isFinite(
      heatIndexC,
    )
  ) {
    return null;
  }

  if (
    heatIndexC <
    HEAT_INDEX_CAUTION_C
  ) {
    return 0.05;
  }

  if (
    heatIndexC <
    HEAT_INDEX_EXTREME_CAUTION_C
  ) {
    const fraction =
      (
        heatIndexC -
        HEAT_INDEX_CAUTION_C
      ) /
      (
        HEAT_INDEX_EXTREME_CAUTION_C -
        HEAT_INDEX_CAUTION_C
      );

    return clamp01(
      0.20 +
        fraction * 0.20,
    );
  }

  if (
    heatIndexC <
    HEAT_INDEX_DANGER_C
  ) {
    const fraction =
      (
        heatIndexC -
        HEAT_INDEX_EXTREME_CAUTION_C
      ) /
      (
        HEAT_INDEX_DANGER_C -
        HEAT_INDEX_EXTREME_CAUTION_C
      );

    return clamp01(
      0.45 +
        fraction * 0.20,
    );
  }

  if (
    heatIndexC <
    HEAT_INDEX_EXTREME_DANGER_C
  ) {
    const fraction =
      (
        heatIndexC -
        HEAT_INDEX_DANGER_C
      ) /
      (
        HEAT_INDEX_EXTREME_DANGER_C -
        HEAT_INDEX_DANGER_C
      );

    return clamp01(
      0.70 +
        fraction * 0.20,
    );
  }

  return 0.95;
}

// ============================================================================
// BASELINE
// ============================================================================

function updateBaseline(
  runtime: RuntimeState,
  ecg: ECGPipelineOutput,
  environment: EnvironmentPipelineOutput,
): void {
  /**
   * IMPORTANT:
   *
   * Only clean physiological/environmental observations should be allowed
   * into the personal baseline.
   *
   * The baseline engine itself also has outlier protection.
   */

  if (
    ecg.sqi.label === "GOOD" ||
    ecg.sqi.label === "USABLE"
  ) {
    runtime.baseline.addObservation(
      "heartRate",
      ecg.heartRate,
    );

    runtime.baseline.addObservations(
      "rrInterval",
      ecg.rrIntervals,
    );

    if (ecg.hrv) {
      runtime.baseline.addObservation(
        "rmssd",
        ecg.hrv.rmssd,
      );

      runtime.baseline.addObservation(
        "sdnn",
        ecg.hrv.sdnn,
      );
    }
  }

  /**
   * Temperature is a legitimate baseline metric in types.ts.
   *
   * Do NOT use AQI as a temperature baseline or vice versa.
   */
  if (
    environment.heatIndexC !== null
  ) {
    runtime.baseline.addObservation(
      "temperature",
      environment.heatIndexC,
    );
  }
}

// ============================================================================
// BASELINE COMPARISON
// ============================================================================

function getBaselineDeviation(
  runtime: RuntimeState,
  metric:
    | "heartRate"
    | "rrInterval"
    | "rmssd"
    | "sdnn"
    | "temperature",
  value: number | null,
): number | undefined {
  const comparison =
    runtime.baseline.compare(
      metric,
      value,
    );

  /**
   * The FalseAlarmGate contract requires a valid baseline deviation
   * before promoting a candidate.
   *
   * undefined deliberately means:
   * "baseline stage has not passed."
   */
  if (
    !comparison.baselineReady ||
    comparison.zScore === null
  ) {
    return undefined;
  }

  return comparison.zScore;
}

// ============================================================================
// CARDIAC CROSS-SENSOR SUPPORT
// ============================================================================

function hasCardiacContextSupport(
  ecg: ECGPipelineOutput,
  ppg: PPGPipelineOutput,
): boolean {
  /**
   * PPG is optional in the current hardware.
   *
   * If it exists and has reasonable confidence, it provides an
   * independent cardiovascular signal.
   */
  if (
    ppg.spo2 !== null &&
    ppg.spo2Confidence >= 0.70
  ) {
    return true;
  }

  /**
   * Without PPG, clean ECG itself remains the primary cardiovascular
   * signal. This is contextual support rather than an independent
   * physiological confirmation.
   */
  return (
    ecg.sqi.label === "GOOD"
  );
}

// ============================================================================
// FUSION
// ============================================================================

function processFusion(
  ecg: ECGPipelineOutput,
  ppg: PPGPipelineOutput,
  motion: MotionPipelineOutput,
  environment: EnvironmentPipelineOutput,
): FusionOutput {
  // --------------------------------------------------------------------------
  // CARDIAC
  //
  // Cardiac candidate comes from persistent rhythm anomaly detection.
  // --------------------------------------------------------------------------

  let cardiacValue:
    number | null = null;

  let cardiacConfidence = 0;

  if (ecg.rhythmFlag) {
    cardiacValue =
      ecg.rhythmFlag.detected
        ? 1
        : 0;

    cardiacConfidence =
      clamp01(
        ecg.rhythmFlag.confidence,
      );
  }

  // --------------------------------------------------------------------------
  // HEAT
  //
  // Fusion expects normalized risk 0..1, NOT heat-index °C.
  // --------------------------------------------------------------------------

  const heatValue =
    calculateHeatRiskScore(
      environment.heatIndexC,
    );

  const heatConfidence =
    heatValue === null
      ? 0
      : environment.heatIndexSource ===
          "unavailable"
        ? 0
        : 1;

  // --------------------------------------------------------------------------
  // RESPIRATORY
  //
  // AQI is first transformed into a normalized environmental respiratory
  // risk score by calculateRespiratoryRisk().
  // --------------------------------------------------------------------------

  const respiratory =
    calculateRespiratoryRisk(
      environment.aqi,
      motion.activityState,
      false,
      environment.aqiConfidence,
    );

  const respiratoryValue =
    respiratory.score;

  const respiratoryConfidence =
    clamp01(
      respiratory.confidence,
    );

  // --------------------------------------------------------------------------
  // FUSION INPUT
  // --------------------------------------------------------------------------

  const fusionInput:
    FusionInput = {
      cardiac: {
        value:
          cardiacValue,

        confidence:
          cardiacConfidence,

        sqi:
          ecg.sqi,
      },

      heat: {
        value:
          heatValue,

        confidence:
          heatConfidence,
      },

      respiratory: {
        value:
          respiratoryValue,

        confidence:
          respiratoryConfidence,
      },

      motion: {
        activityState:
          motion.activityState,

        fall:
          motion.fall,
      },
    };

  return fuseSensorRisks(
    fusionInput,
  );
}

// ============================================================================
// FALSE-ALARM GATE
// ============================================================================

function evaluateFalseAlarmGates(
  runtime: RuntimeState,
  fusion: FusionOutput,
  ecg: ECGPipelineOutput,
  motion: MotionPipelineOutput,
  cardiacBaselineDeviation:
    number | undefined,
  heatBaselineDeviation:
    number | undefined,
): {
  cardiacGate: ReturnType<
    RuntimeState["falseAlarmGate"]["evaluate"]
  >;

  heatGate: ReturnType<
    RuntimeState["falseAlarmGate"]["evaluate"]
  >;

  respiratoryGate: ReturnType<
    RuntimeState["falseAlarmGate"]["evaluate"]
  >;

  fallGate: ReturnType<
    RuntimeState["falseAlarmGate"]["evaluate"]
  >;
} {
  // --------------------------------------------------------------------------
  // CARDIAC
  // --------------------------------------------------------------------------

  const cardiacInput:
    FalseAlarmGateInput = {
      category: "cardiac",

      candidateScore:
        fusion.cardiac.score ?? 0,

      sqi:
        ecg.sqi,

      activityState:
        motion.activityState,

      baselineDeviation:
        cardiacBaselineDeviation,

      crossSensorSupport:
        hasCardiacContextSupport(
          ecg,
          {
            spo2: null,
            spo2Confidence: 0,
            ppgQuality: "UNAVAILABLE",
            ppgDerivedHR: null,
          },
        ),

      confidence:
        fusion.cardiac.confidence,
    };

  const cardiacGate =
    runtime.falseAlarmGate.evaluate(
      cardiacInput,
      Date.now(),
    );

  // --------------------------------------------------------------------------
  // HEAT
  // --------------------------------------------------------------------------

  const heatInput:
    FalseAlarmGateInput = {
      category: "heat",

      candidateScore:
        fusion.heat.score ?? 0,

      activityState:
        motion.activityState,

      baselineDeviation:
        heatBaselineDeviation,

      confidence:
        fusion.heat.confidence,
    };

  const heatGate =
    runtime.falseAlarmGate.evaluate(
      heatInput,
      Date.now(),
    );

  // --------------------------------------------------------------------------
  // RESPIRATORY
  //
  // IMPORTANT CONTRACT LIMITATION:
  //
  // types.ts has no personal AQI/environment baseline metric.
  //
  // We therefore intentionally pass undefined rather than pretending
  // that temperature deviation is an AQI baseline.
  //
  // The existing FalseAlarmGate consequently blocks alert promotion
  // until the architecture gains an appropriate environmental baseline
  // contract.
  // --------------------------------------------------------------------------

  const respiratoryInput:
    FalseAlarmGateInput = {
      category: "respiratory",

      candidateScore:
        fusion.respiratory.score ?? 0,

      activityState:
        motion.activityState,

      baselineDeviation:
        undefined,

      confidence:
        fusion.respiratory.confidence,
    };

  const respiratoryGate =
    runtime.falseAlarmGate.evaluate(
      respiratoryInput,
      Date.now(),
    );

  // --------------------------------------------------------------------------
  // FALL
  //
  // Same architectural limitation:
  //
  // types.ts has no fall-specific personal baseline metric.
  //
  // We intentionally do NOT fabricate a baseline deviation from
  // acceleration/confidence.
  // --------------------------------------------------------------------------

  const fallInput:
    FalseAlarmGateInput = {
      category: "fall",

      candidateScore:
        fusion.fall.detected
          ? 1
          : 0,

      activityState:
        motion.activityState,

      baselineDeviation:
        undefined,

      confidence:
        fusion.fall.confidence,
    };

  const fallGate =
    runtime.falseAlarmGate.evaluate(
      fallInput,
      Date.now(),
    );

  return {
    cardiacGate,
    heatGate,
    respiratoryGate,
    fallGate,
  };
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Main Sanjeevni AI entry point.
 *
 * React Native should call this after its Bluetooth ingestion layer
 * has assembled a normalized SensorTickInput.
 *
 * The function is synchronous because Phase 1 AI is entirely local.
 */
export function processSensorTick(
  input: SensorTickInput,
): SanjeevniRiskOutput {
  // --------------------------------------------------------------------------
  // DEFENSIVE VALIDATION
  // --------------------------------------------------------------------------

  if (
    !input ||
    typeof input.userId !== "string" ||
    input.userId.trim().length === 0 ||
    !isFiniteNumber(
      input.timestamp,
    )
  ) {
    return {
      timestamp:
        isFiniteNumber(
          input?.timestamp,
        )
          ? input.timestamp
          : 0,

      status:
        "initializing",

      heartRate:
        null,

      hrv:
        null,

      spo2: {
        value:
          null,

        confidence:
          0,
      },

      environment: {
        temperature:
          null,

        humidity:
          null,

        aqi:
          null,

        heatIndex:
          null,
      },

      motion: {
        level:
          0,

        state:
          "REST",
      },

      signalQuality: {
        ecg:
          0,

        ppg:
          0,
      },

      risks: {
        cardiac: {
          level:
            "NORMAL",

          score:
            0,

          confidence:
            0,

          evidence: [
            "Invalid sensor input",
          ],
        },

        heat: {
          level:
            "NORMAL",

          score:
            0,

          confidence:
            0,

          evidence: [
            "Invalid sensor input",
          ],
        },

        respiratory: {
          level:
            "NORMAL",

          score:
            0,

          confidence:
            0,

          evidence: [
            "Invalid sensor input",
          ],
        },

        fall: {
          detected:
            false,

          confidence:
            0,
        },
      },

      sosRecommended:
        false,
    };
  }

  const runtime =
    getRuntime(
      input.userId,
    );

  // --------------------------------------------------------------------------
  // TIMESTAMP ORDERING
  // --------------------------------------------------------------------------

  const outOfOrder =
    runtime.lastTimestamp !== null &&
    input.timestamp <
      runtime.lastTimestamp;

  /**
   * Never allow stale packets to mutate:
   * - filter state
   * - RR history
   * - baseline
   * - false-alarm persistence
   */
  if (outOfOrder) {
    return {
      timestamp:
        input.timestamp,

      status:
        runtime.initialized
          ? "ready"
          : "initializing",

      heartRate:
        null,

      hrv:
        null,

      spo2: {
        value:
          null,

        confidence:
          0,
      },

      environment: {
        temperature:
          null,

        humidity:
          null,

        aqi:
          null,

        heatIndex:
          null,
      },

      motion: {
        level:
          0,

        state:
          "REST",
      },

      signalQuality: {
        ecg:
          0,

        ppg:
          0,
      },

      risks: {
        cardiac: {
          level:
            "NORMAL",

          score:
            0,

          confidence:
            0,

          evidence: [
            "Out-of-order sensor tick ignored",
          ],
        },

        heat: {
          level:
            "NORMAL",

          score:
            0,

          confidence:
            0,

          evidence: [
            "Out-of-order sensor tick ignored",
          ],
        },

        respiratory: {
          level:
            "NORMAL",

          score:
            0,

          confidence:
            0,

          evidence: [
            "Out-of-order sensor tick ignored",
          ],
        },

        fall: {
          detected:
            false,

          confidence:
            0,
        },
      },

      sosRecommended:
        false,
    };
  }

  runtime.lastTimestamp =
    input.timestamp;

  runtime.initialized =
    true;

  // --------------------------------------------------------------------------
  // 1. MOTION
  //
  // Motion is processed first because ECG SQI depends on motionLevel.
  // --------------------------------------------------------------------------

  const motion =
    processMotion(
      input,
    );

  // --------------------------------------------------------------------------
  // 2. ECG
  // --------------------------------------------------------------------------

  const ecg =
    processECG(
      input,
      runtime,
      motion,
    );

  // --------------------------------------------------------------------------
  // 3. PPG
  // --------------------------------------------------------------------------

  const ppg =
    processPPG(
      input,
      motion,
    );

  // --------------------------------------------------------------------------
  // 4. ENVIRONMENT
  // --------------------------------------------------------------------------

  const environment =
    processEnvironment(
      input,
    );

  // --------------------------------------------------------------------------
  // 5. BASELINE COMPARISON
  //
  // MUST happen before baseline update.
  // --------------------------------------------------------------------------

  const cardiacBaselineDeviation =
    getBaselineDeviation(
      runtime,
      "heartRate",
      ecg.heartRate,
    );

  const heatBaselineDeviation =
    getBaselineDeviation(
      runtime,
      "temperature",
      environment.heatIndexC,
    );

  // --------------------------------------------------------------------------
  // 6. FUSION
  // --------------------------------------------------------------------------

  const fusion =
    processFusion(
      ecg,
      ppg,
      motion,
      environment,
    );

  // --------------------------------------------------------------------------
  // 7. FALSE-ALARM GATES
  //
  // IMPORTANT:
  //
  // evaluate() requires the actual analysis timestamp.
  // --------------------------------------------------------------------------

  const cardiacGate =
    runtime.falseAlarmGate.evaluate(
      {
        category:
          "cardiac",

        candidateScore:
          fusion.cardiac.score ?? 0,

        sqi:
          ecg.sqi,

        activityState:
          motion.activityState,

        baselineDeviation:
          cardiacBaselineDeviation,

        crossSensorSupport:
          hasCardiacContextSupport(
            ecg,
            ppg,
          ),

        confidence:
          fusion.cardiac.confidence,
      },
      input.timestamp,
    );

  const heatGate =
    runtime.falseAlarmGate.evaluate(
      {
        category:
          "heat",

        candidateScore:
          fusion.heat.score ?? 0,

        activityState:
          motion.activityState,

        baselineDeviation:
          heatBaselineDeviation,

        confidence:
          fusion.heat.confidence,
      },
      input.timestamp,
    );

  /**
   * No temperature baseline is substituted for AQI.
   *
   * This is deliberately conservative because types.ts does not define
   * an environmental/AQI personal baseline.
   */
  const respiratoryGate =
    runtime.falseAlarmGate.evaluate(
      {
        category:
          "respiratory",

        candidateScore:
          fusion.respiratory.score ?? 0,

        activityState:
          motion.activityState,

        baselineDeviation:
          undefined,

        confidence:
          fusion.respiratory.confidence,
      },
      input.timestamp,
    );

  /**
   * No fake baseline value is supplied for falls.
   *
   * The current types.ts has no fall baseline metric.
   */
  const fallGate =
    runtime.falseAlarmGate.evaluate(
      {
        category:
          "fall",

        candidateScore:
          fusion.fall.detected
            ? 1
            : 0,

        activityState:
          motion.activityState,

        baselineDeviation:
          undefined,

        confidence:
          fusion.fall.confidence,
      },
      input.timestamp,
    );

  // --------------------------------------------------------------------------
  // 8. RISK DECISION ENGINE
  // --------------------------------------------------------------------------

  const cardiacDecision =
    runtime.riskDecisionEngine.decideCardiac(
      fusion.cardiac.score ?? 0,

      fusion.cardiac.confidence,

      cardiacGate,

      input.timestamp,

      fusion.cardiac.evidence,
    );

  const heatDecision =
    runtime.riskDecisionEngine.decideHeat(
      fusion.heat.score ?? 0,

      fusion.heat.confidence,

      heatGate,

      input.timestamp,

      fusion.heat.evidence,
    );

  const respiratoryDecision =
    runtime.riskDecisionEngine.decideRespiratory(
      fusion.respiratory.score ?? 0,

      fusion.respiratory.confidence,

      respiratoryGate,

      input.timestamp,

      fusion.respiratory.evidence,
    );

  const fallDecision =
    runtime.riskDecisionEngine.decideFall(
      fusion.fall.detected
        ? 1
        : 0,

      fusion.fall.confidence,

      fallGate,

      input.timestamp,

      fusion.fall.detected
        ? [
            "Fall-like motion event detected",
          ]
        : [],
    );

  // --------------------------------------------------------------------------
  // 9. BASELINE UPDATE
  //
  // Deliberately AFTER comparison and decision.
  //
  // This prevents the current observation from changing the baseline
  // before its own anomaly evaluation.
  // --------------------------------------------------------------------------

  updateBaseline(
    runtime,
    ecg,
    environment,
  );

  // --------------------------------------------------------------------------
  // 10. SOS
  //
  // AI recommends only.
  //
  // React Native/mobile layer performs the actual emergency action.
  // --------------------------------------------------------------------------

  const sosRecommended =
    cardiacDecision.sosRecommended ||
    fallDecision.sosRecommended;

  // --------------------------------------------------------------------------
  // 11. STATUS
  //
  // "ready" means the orchestration engine is operational.
  // It does NOT mean every sensor or every baseline is ready.
  // --------------------------------------------------------------------------

  const status:
    "ready" | "initializing" =
    runtime.initialized
      ? "ready"
      : "initializing";

  // --------------------------------------------------------------------------
  // 12. PUBLIC OUTPUT
  // --------------------------------------------------------------------------

  return {
    timestamp:
      input.timestamp,

    status,

    heartRate:
      ecg.heartRate,

    hrv:
      ecg.hrv
        ? {
            rmssd:
              ecg.hrv.rmssd,

            sdnn:
              ecg.hrv.sdnn,
          }
        : null,

    spo2: {
      value:
        ppg.spo2,

      confidence:
        clamp01(
          ppg.spo2Confidence,
        ),
    },

    environment: {
      temperature:
        input.environment?.tempC ??
        null,

      humidity:
        input.environment?.humidityPct ??
        null,

      aqi:
        environment.aqi,

      heatIndex:
        environment.heatIndexC,
    },

    motion: {
      level:
        clamp01(
          motion.motionLevel,
        ),

      state:
        motion.activityState,
    },

    signalQuality: {
      ecg:
        clamp01(
          ecg.sqi.score,
        ),

      ppg:
        clamp01(
          ppg.spo2Confidence,
        ),
    },

    risks: {
      cardiac: {
        level:
          cardiacDecision.level,

        score:
          cardiacDecision.score,

        confidence:
          cardiacDecision.confidence,

        evidence:
          cardiacDecision.evidence,
      },

      heat: {
        level:
          heatDecision.level,

        score:
          heatDecision.score,

        confidence:
          heatDecision.confidence,

        evidence:
          heatDecision.evidence,
      },

      respiratory: {
        level:
          respiratoryDecision.level,

        score:
          respiratoryDecision.score,

        confidence:
          respiratoryDecision.confidence,

        evidence:
          respiratoryDecision.evidence,
      },

      fall: {
        detected:
          fusion.fall.detected,

        confidence:
          clamp01(
            fusion.fall.confidence,
          ),
      },
    },

    sosRecommended,
  };
}

// ============================================================================
// RESET
// ============================================================================

/**
 * Reset the entire AI engine.
 *
 * Useful for:
 * - application logout/login
 * - user/device reassignment
 * - sensor replacement
 * - testing
 *
 * Since all state is now per-user, clearing this map resets every user.
 */
export function resetAIEngine(): void {
  runtimeByUser.clear();
}

/**
 * Reset only one user's AI session.
 *
 * This is useful when the same application remains logged in but
 * the wearable is disconnected/reassigned.
 */
export function resetAIEngineForUser(
  userId: string,
): void {
  if (
    typeof userId !== "string" ||
    userId.trim().length === 0
  ) {
    return;
  }

  runtimeByUser.delete(
    userId,
  );
}
