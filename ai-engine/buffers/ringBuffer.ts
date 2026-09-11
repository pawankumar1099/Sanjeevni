// ============================================================================
// ai-engine/buffers/ringBuffer.ts
// A generic, fixed-capacity ring buffer for streaming numeric data.
// Used by ECG/PPG/motion pipelines to maintain a sliding window of recent
// samples without reallocating arrays or recomputing from scratch every tick.
// ============================================================================

export class RingBuffer<T> {
  private buffer: (T | undefined)[];
  private capacity: number;
  private writeIndex: number = 0;
  private count: number = 0; // how many valid elements currently held (<= capacity)

  constructor(capacity: number) {
    if (capacity <= 0) {
      throw new Error("RingBuffer capacity must be a positive integer");
    }
    this.capacity = capacity;
    this.buffer = new Array(capacity).fill(undefined);
  }

  /**
   * Push a new value into the buffer. Oldest value is overwritten
   * automatically once capacity is reached — this is the core
   * "sliding window" behavior.
   */
  push(value: T): void {
    this.buffer[this.writeIndex] = value;
    this.writeIndex = (this.writeIndex + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count += 1;
    }
  }

  /**
   * Push multiple values at once (e.g. an incoming chunk of ECG samples).
   */
  pushMany(values: T[]): void {
    for (const v of values) {
      this.push(v);
    }
  }

  /**
   * Returns all currently held values in correct chronological order
   * (oldest first, newest last). Safe to call even if buffer isn't full yet.
   */
  toArray(): T[] {
    if (this.count < this.capacity) {
      // Buffer not yet full: valid data is from index 0 to writeIndex-1
      return this.buffer.slice(0, this.count) as T[];
    }
    // Buffer full: valid data wraps around, starting at writeIndex
    return [
      ...this.buffer.slice(this.writeIndex),
      ...this.buffer.slice(0, this.writeIndex),
    ] as T[];
  }

  /**
   * Returns the most recent N values (newest last). Useful for rolling
   * windows like "last 5 RR intervals" without pulling the whole buffer.
   */
  lastN(n: number): T[] {
    const all = this.toArray();
    if (n >= all.length) return all;
    return all.slice(all.length - n);
  }

  /**
   * True once the buffer has been filled at least once.
   * Pipelines should treat data as "not ready" (avoid emitting
   * confident output) until this is true — this is how we implement
   * the "startup transient" guard from Part 6.1.
   */
  isFull(): boolean {
    return this.count === this.capacity;
  }

  /**
   * How many valid elements are currently held.
   */
  size(): number {
    return this.count;
  }

  /**
   * Clears the buffer entirely — used when a sensor disconnects/reconnects
   * and old data would no longer be valid to mix with new data.
   */
  clear(): void {
    this.buffer = new Array(this.capacity).fill(undefined);
    this.writeIndex = 0;
    this.count = 0;
  }

  getCapacity(): number {
    return this.capacity;
  }
}