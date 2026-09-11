import * as SQLite from 'expo-sqlite';
import { Platform } from 'react-native';

// Singleton connection — reused across all functions to avoid
// race conditions between BLE writes and dashboard reads.
let dbInstance: SQLite.SQLiteDatabase | null = null;
// Promise lock: prevents multiple concurrent callers from each calling
// openDatabaseAsync() simultaneously (which causes a NullPointerException
// in the Android native layer). All concurrent calls await the same promise.
let dbOpenPromise: Promise<SQLite.SQLiteDatabase> | null = null;

export async function getDb(): Promise<SQLite.SQLiteDatabase> {
  if (Platform.OS === 'web') {
    // Return a dummy mock object for web so the UI doesn't crash
    // Web requires complex WASM workers for SQLite, so we mock it for rapid UI testing.
    return {
      execAsync: async () => {},
      runAsync: async () => {},
      getFirstAsync: async () => null,
      getAllAsync: async () => []
    } as unknown as SQLite.SQLiteDatabase;
  }

  // Return cached instance immediately if already open
  if (dbInstance) return dbInstance;

  // If an open is already in progress, wait for it instead of starting another
  if (!dbOpenPromise) {
    dbOpenPromise = SQLite.openDatabaseAsync('sanjeevni.db').then(db => {
      dbInstance = db;
      dbOpenPromise = null; // Reset lock after successful open
      return db;
    });
  }
  return dbOpenPromise;
}

/**
 * Initializes the Sanjeevni on-device SQLite database according to the SIH Build Guide.
 */
export async function initDb() {
  const db = await getDb();

  // Always ensure tables exist (CREATE TABLE IF NOT EXISTS is idempotent).
  // This prevents "no such table" errors if a previous init was interrupted
  // or the DB file exists but is empty/corrupt.
  // ⚠️  MIGRATION NOTE: If you add a column here, also add an ALTER TABLE
  //    migration in the versioned block below. Both must stay in sync.
  await db.execAsync(`
    CREATE TABLE IF NOT EXISTS readings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      sensor_type TEXT NOT NULL,
      value REAL NOT NULL,
      source TEXT NOT NULL,
      synced INTEGER DEFAULT 0
    );

    -- Gap #4: Composite index for faster dashboard queries
    CREATE INDEX IF NOT EXISTS idx_readings_sensor_timestamp ON readings(sensor_type, timestamp);

    CREATE TABLE IF NOT EXISTS alerts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      category TEXT NOT NULL,
      -- Gap #3: Strict severity enum constraint
      severity TEXT CHECK(severity IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')) NOT NULL,
      message TEXT NOT NULL,
      acknowledged INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);

    CREATE TABLE IF NOT EXISTS user_profile (
      -- Gap #1: Enforce single profile row at DB level
      id INTEGER PRIMARY KEY CHECK(id = 1),
      baseline_values TEXT,
      vulnerability_flags TEXT,
      emergency_contact TEXT
    );

    CREATE TABLE IF NOT EXISTS rolling_baseline (
      -- Gap #1: metric_type as PRIMARY KEY for ON CONFLICT DO UPDATE
      metric_type TEXT PRIMARY KEY,
      moving_average REAL NOT NULL,
      last_updated TEXT NOT NULL
    );
  `);

  // Gap #9: Schema Versioning Strategy — track version for future migrations
  // ⚠️  MIGRATION NOTE: When adding a new migration block (e.g. v3), also
  //    update the CREATE TABLE IF NOT EXISTS block above for fresh installs.
  const result = await db.getFirstAsync<{user_version: number}>('PRAGMA user_version');
  const currentVersion = result ? result.user_version : 0;

  if (currentVersion < 1) {
    await db.execAsync('PRAGMA user_version = 1');
    console.log('Database migrated to v1 schema.');
  }

  if (currentVersion < 2) {
    // v2 migration: add `synced` column to existing readings tables that were
    // created before this column existed. ALTER TABLE ADD COLUMN is safe on
    // SQLite — we wrap in try/catch because it throws if the column already
    // exists (e.g. on a clean install where CREATE TABLE already included it).
    try {
      await db.execAsync('ALTER TABLE readings ADD COLUMN synced INTEGER DEFAULT 0');
      console.log('Database migrated to v2: added synced column.');
    } catch {
      // Column already exists — no action needed
    }
    await db.execAsync('PRAGMA user_version = 2');
  } else {
    console.log('Database schema is already up to date.');
  }
}

/**
 * P2: Prunes raw readings older than 7 days to prevent DB bloat.
 * Should be called periodically (e.g., on app startup or daily cron).
 */
export async function pruneOldData() {
  const db = await getDb();

  const date7DaysAgo = new Date();
  date7DaysAgo.setDate(date7DaysAgo.getDate() - 7);
  const cutoffTimestamp = date7DaysAgo.toISOString();

  await db.runAsync(
    'DELETE FROM readings WHERE timestamp < ?',
    cutoffTimestamp
  );

  console.log(`Pruned readings older than ${cutoffTimestamp}`);
}

// ---------------- READINGS ----------------

/**
 * Insert a single reading into the unified table.
 */
export async function insertReading(sensorType: string, value: number, source: string = 'garment') {
  // Gap #7: Data Quality - Validate incoming sensor values to prevent false AI alerts.
  // Loose bounds catch disconnected/erroring sensors (e.g. DHT22 returns -999, MQ135 spikes)
  // without rejecting legitimate edge-case readings.
  if (sensorType === 'HR' && (value <= 0 || value > 250)) {
    console.warn(`Rejected invalid HR reading: ${value}`);
    return;
  }
  if (sensorType === 'SpO2' && (value < 0 || value > 100)) {
    console.warn(`Rejected invalid SpO2 reading: ${value}`);
    return;
  }
  if (sensorType === 'TEMP' && (value < -10 || value > 60)) {
    console.warn(`Rejected invalid TEMP reading: ${value}`);
    return;
  }
  if (sensorType === 'AQI' && (value < 0 || value > 1000)) {
    console.warn(`Rejected invalid AQI reading: ${value}`);
    return;
  }
  if (sensorType === 'HUMIDITY' && (value < 0 || value > 100)) {
    console.warn(`Rejected invalid HUMIDITY reading: ${value}`);
    return;
  }
  if (sensorType === 'STEPS' && (value < 0 || value > 100000)) {
    console.warn(`Rejected invalid STEPS reading: ${value}`);
    return;
  }
  if (sensorType === 'HRV_SDNN' && (value < 0 || value > 500)) {
    console.warn(`Rejected invalid HRV_SDNN reading: ${value}`);
    return;
  }
  if (sensorType === 'HRV_RMSSD' && (value < 0 || value > 500)) {
    console.warn(`Rejected invalid HRV_RMSSD reading: ${value}`);
    return;
  }

  const db = await getDb();
  const timestamp = new Date().toISOString();

  await db.runAsync(
    'INSERT INTO readings (timestamp, sensor_type, value, source, synced) VALUES (?, ?, ?, ?, 0)',
    timestamp, sensorType, value, source
  );
}

/**
 * Gap #5: Sync logic - Get all readings that haven't been backed up to the cloud.
 */
export async function getPendingReadings(): Promise<any[]> {
  const db = await getDb();
  return db.getAllAsync('SELECT * FROM readings WHERE synced = 0');
}

/**
 * Gap #5: Sync logic - Mark specific readings as successfully synced.
 */
export async function markSynced(ids: number[]) {
  if (ids.length === 0) return;
  const db = await getDb();
  const placeholders = ids.map(() => '?').join(',');
  await db.runAsync(`UPDATE readings SET synced = 1 WHERE id IN (${placeholders})`, ...ids);
}

/**
 * Fetch the latest reading for a specific sensor type.
 */
export async function getLatestReading(sensorType: string): Promise<any> {
  const db = await getDb();
  const result = await db.getFirstAsync(
    'SELECT * FROM readings WHERE sensor_type = ? ORDER BY timestamp DESC LIMIT 1',
    sensorType
  );
  return result;
}

/**
 * Fetch all readings for a sensor type within the last N minutes.
 * Used for the dashboard trend view / "last N minutes" queries.
 */
export async function getRecentReadings(sensorType: string, minutes: number): Promise<any[]> {
  const db = await getDb();
  const cutoff = new Date(Date.now() - minutes * 60 * 1000).toISOString();

  return db.getAllAsync(
    'SELECT * FROM readings WHERE sensor_type = ? AND timestamp >= ? ORDER BY timestamp DESC',
    sensorType, cutoff
  );
}

// ---------------- ALERTS ----------------

/**
 * Insert a new alert (e.g. heat risk, fall detected, abnormal HR).
 */
export async function insertAlert(category: string, severity: string, message: string) {
  const db = await getDb();
  const timestamp = new Date().toISOString();

  await db.runAsync(
    'INSERT INTO alerts (timestamp, category, severity, message) VALUES (?, ?, ?, ?)',
    timestamp, category, severity, message
  );
}

/**
 * Fetch recent alerts, most recent first, for the alert banner / history feed.
 */
export async function getAlerts(limit: number = 20): Promise<any[]> {
  const db = await getDb();
  return db.getAllAsync(
    'SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?',
    limit
  );
}

/**
 * Mark an alert as acknowledged (e.g. user dismissed it or cancelled auto-SOS).
 */
export async function acknowledgeAlert(id: number) {
  const db = await getDb();
  await db.runAsync(
    'UPDATE alerts SET acknowledged = 1 WHERE id = ?',
    id
  );
}

// ---------------- USER PROFILE ----------------

/**
 * Fetch the single user profile row (baseline values, vulnerability flags, emergency contact).
 */
export async function getUserProfile(): Promise<any> {
  const db = await getDb();
  return db.getFirstAsync('SELECT * FROM user_profile LIMIT 1');
}

/**
 * Create or update the user profile. Used for onboarding / settings screen,
 * and read by the SOS flow for the emergency contact.
 */
export async function upsertUserProfile(
  baselineValues: string,
  vulnerabilityFlags: string,
  emergencyContact: string
) {
  const db = await getDb();
  const existing = await getUserProfile();

  if (existing) {
    await db.runAsync(
      'UPDATE user_profile SET baseline_values = ?, vulnerability_flags = ?, emergency_contact = ? WHERE id = ?',
      baselineValues, vulnerabilityFlags, emergencyContact, existing.id
    );
  } else {
    await db.runAsync(
      'INSERT INTO user_profile (baseline_values, vulnerability_flags, emergency_contact) VALUES (?, ?, ?)',
      baselineValues, vulnerabilityFlags, emergencyContact
    );
  }
}

// ---------------- ROLLING BASELINE ----------------

/**
 * P2: Updates the AI rolling baseline for a specific metric.
 */
export async function updateRollingBaseline(metricType: string, movingAverage: number) {
  const db = await getDb();
  const timestamp = new Date().toISOString();

  await db.runAsync(
    `INSERT INTO rolling_baseline (metric_type, moving_average, last_updated) 
      VALUES (?, ?, ?) 
      ON CONFLICT(metric_type) 
      DO UPDATE SET moving_average=excluded.moving_average, last_updated=excluded.last_updated`,
    metricType, movingAverage, timestamp
  );
}

/**
 * Fetch the current rolling baseline for a metric (used by the AI person's
 * personalization logic to compare live readings against).
 */
export async function getRollingBaseline(metricType: string): Promise<any> {
  const db = await getDb();
  return db.getFirstAsync(
    'SELECT * FROM rolling_baseline WHERE metric_type = ?',
    metricType
  );
}