// ============================================================================
// ai-engine/types.ts
// Central type contracts for the Sanjeevni on-device AI engine.
// Every module's public function signatures reference these types.
// Changing a type here is a breaking change across the whole engine —
// treat this file as the "API surface" of the entire ai-engine folder.
// ============================================================================

// ---------------------------------------------------------------------------
// Common primitives
// ---------------------------------------------------------------------------

export type Timestamp = number; // epoch millis

export type SQILabel = "GOOD" | "USABLE" | "NOISY" | "MOTION_CORRUPTED" | "INVALID";

export interface SQI {
  label: SQILabel;
  score: number; // 0.0 - 1.0
}

export type ActivityState = "REST" | "LIGHT" | "ACTIVE";

export type RiskLevel = "NORMAL" | "CAUTION" | "RISK";

export type DataSource = "wearable" | "wearable_proxy" | "api" | "fused" | "unavailable";

// ---------------------------------------------------------------------------
// RAW SENSOR INPUT — what comes off BLE, already parsed into numbers
// ---------------------------------------------------------------------------

export interface ECGSample {
  samples: number[];         // raw ECG amplitude values for this chunk
  sampleRate: number;        // Hz, expected 250-360
  timestampStart: Timestamp;
  timestampEnd: Timestamp;
}

export interface PPGSample {
  irSamples: number[];
  redSamples: number[];
  sampleRate: number;
  timestamp: Timestamp;
}

export interface MotionSample {
  accel: { x: number; y: number; z: number }[];
  gyro: { x: number; y: number; z: number }[];
  sampleRate: number; // Hz, expected 50-100
  timestamp: Timestamp;
}

export interface EnvironmentSample {
  tempC: number | null;
  humidityPct: number | null;
  mq135Raw: number | null;
  apiAQI: number | null;
  apiWeather: { tempC?: number; humidityPct?: number } | null;
  timestamp: Timestamp;
}

// Full raw packet for one "tick" of processing (what processSensorTick receives)
export interface SensorTickInput {
  ecg?: ECGSample;
  ppg?: PPGSample;
  motion?: MotionSample;
  environment?: EnvironmentSample;
  userId: string; // needed for personal baseline lookups
  timestamp: Timestamp;
}

// ---------------------------------------------------------------------------
// ECG PIPELINE outputs (Part 5.1 / Part 6)
// ---------------------------------------------------------------------------

export interface FilteredECG {
  filteredSamples: number[];
  sampleRate: number;
}

export interface RPeakResult {
  peakIndices: number[];       // sample indices of detected R-peaks
  peakTimestamps: Timestamp[]; // absolute timestamps of each peak
}

export interface RRResult {
  rrIntervalsMs: number[];   // validated RR intervals only
  rejectedCount: number;     // how many candidate RR were thrown out (for debugging/QA)
}

export interface HRVResult {
  sdnn: number | null;
  rmssd: number | null;
  pnn50: number | null;
  pnn50Confidence: number; // low if window too short (Part 6.6)
}

export interface RhythmFlag {
  detected: boolean;
  confidence: number; // 0.0-1.0
  evidence: string[];
}

export interface ECGPipelineOutput {
  heartRate: number | null;
  rrIntervals: number[];
  hrv: HRVResult | null;
  rhythmFlag: RhythmFlag | null;
  sqi: SQI;
}

// ---------------------------------------------------------------------------
// PPG / SpO2 PIPELINE outputs (Part 5.2)
// ---------------------------------------------------------------------------

export type PPGQuality = "GOOD" | "POOR" | "UNAVAILABLE";

export interface PPGPipelineOutput {
  spo2: number | null;
  spo2Confidence: number; // 0 if unavailable
  ppgQuality: PPGQuality;
  ppgDerivedHR: number | null; // cross-check only, never primary
}

// ---------------------------------------------------------------------------
// MOTION PIPELINE outputs (Part 5.3)
// ---------------------------------------------------------------------------

export type FallStage = "NONE" | "FREEFALL" | "IMPACT" | "CONFIRMED";

export interface FallResult {
  detected: boolean;
  confidence: number;
  stage: FallStage;
}

export interface MotionPipelineOutput {
  motionLevel: number; // normalized magnitude, used for gating other sensors
  activityState: ActivityState;
  fall: FallResult;
}

// ---------------------------------------------------------------------------
// ENVIRONMENT PIPELINE outputs (Part 5.4)
// ---------------------------------------------------------------------------

export interface EnvironmentPipelineOutput {
  heatIndexC: number | null;
  heatIndexSource: DataSource;
  aqi: number | null;
  aqiSource: DataSource;
  aqiConfidence: number;
}

// ---------------------------------------------------------------------------
// PERSONAL BASELINE (Part 22 / baseline/personalBaseline.ts)
// ---------------------------------------------------------------------------

export interface BaselineEntry {
  metric: string;         // e.g. "restingHR", "rrVariance"
  baseline: number;
  variance: number;
  sampleCount: number;
  updatedAt: Timestamp;
}

export interface BaselineLookupResult {
  hasSufficientHistory: boolean; // false during cold-start
  entry: BaselineEntry | null;
}

// ---------------------------------------------------------------------------
// FUSION LAYER (Part 8)
// ---------------------------------------------------------------------------

export interface SensorConfidenceInput {
  value: number | null;
  confidence: number; // 0.0-1.0
}

export interface FusionInput {
  cardiac: SensorConfidenceInput & { sqi: SQI };
  heat: SensorConfidenceInput;
  respiratory: SensorConfidenceInput;
  motion: { activityState: ActivityState; fall: FallResult };
}

export interface FusedCategoryRisk {
  score: number | null;       // null if INSUFFICIENT_DATA
  confidence: number;
  evidence: string[];
  insufficientData: boolean;
}

export interface FusionOutput {
  cardiac: FusedCategoryRisk;
  heat: FusedCategoryRisk;
  respiratory: FusedCategoryRisk;
  fall: FallResult;
}

// ---------------------------------------------------------------------------
// FALSE-ALARM GATE (Part 9)
// ---------------------------------------------------------------------------

export type SuppressReason =
  | "LOW_QUALITY"
  | "MOTION"
  | "INSUFFICIENT_PERSISTENCE"
  | "LOW_CONFIDENCE"
  | "COOLDOWN"
  | null;

export interface FalseAlarmGateInput {
  category: "cardiac" | "heat" | "respiratory" | "fall";
  candidateScore: number;
  sqi?: SQI;
  activityState?: ActivityState;
  baselineDeviation?: number;
  crossSensorSupport?: boolean;
  confidence: number;
}

export interface FalseAlarmGateOutput {
  shouldAlert: boolean;
  suppressedReason: SuppressReason;
  evidence: string[];
}

// ---------------------------------------------------------------------------
// DECISION ENGINE (Part 24) — final per-category risk object
// ---------------------------------------------------------------------------

export interface CategoryRiskDecision {
  category: "cardiac" | "heat" | "respiratory" | "fall";
  level: RiskLevel;
  score: number;
  confidence: number;
  evidence: string[];
  timestamp: Timestamp;
  recommendedAction: string;
  sosRecommended: boolean;
}

// ---------------------------------------------------------------------------
// FINAL PUBLIC OUTPUT — the exact JSON contract from Part 11
// This is what processSensorTick() returns to the React Native app.
// ---------------------------------------------------------------------------

export interface SanjeevniRiskOutput {
  timestamp: Timestamp;
  status: "ready" | "initializing";
  heartRate: number | null;
  hrv: { rmssd: number | null; sdnn: number | null } | null;
  spo2: { value: number | null; confidence: number };
  environment: {
    temperature: number | null;
    humidity: number | null;
    aqi: number | null;
    heatIndex: number | null;
  };
  motion: { level: number; state: ActivityState };
  signalQuality: { ecg: number; ppg: number };
  risks: {
    cardiac: { level: RiskLevel; score: number; confidence: number; evidence: string[] };
    heat: { level: RiskLevel; score: number; confidence: number; evidence: string[] };
    respiratory: { level: RiskLevel; score: number; confidence: number; evidence: string[] };
    fall: { detected: boolean; confidence: number };
  };
  sosRecommended: boolean;
}