import "@testing-library/jest-dom/vitest";

/**
 * Node 22+'s own experimental global `localStorage` shadows jsdom's real
 * implementation and throws without a `--localstorage-file` flag --
 * unrelated to anything the app does, since every real browser has a
 * working `localStorage`. A small in-memory shim keeps zustand's
 * `persist` middleware (used by every store) functional in tests.
 */
class InMemoryStorage implements Storage {
  private store = new Map<string, string>();

  get length(): number {
    return this.store.size;
  }

  clear(): void {
    this.store.clear();
  }

  getItem(key: string): string | null {
    return this.store.has(key) ? this.store.get(key)! : null;
  }

  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }

  setItem(key: string, value: string): void {
    this.store.set(key, value);
  }
}

Object.defineProperty(globalThis, "localStorage", {
  value: new InMemoryStorage(),
  configurable: true,
  writable: true,
});

// jsdom implements neither scrollIntoView nor scrolling in general; the
// app calls it for evidence-card/PDF-highlight synchronization. A no-op
// shim keeps component behavior testable without faking scroll positions.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
