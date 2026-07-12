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

// jsdom has no Web Animations API either; Base UI's scroll-area calls
// `getAnimations()` to decide when the auto-hide scrollbar has finished
// fading. Every real browser implements this -- the shim just reports
// "nothing running", which is always true in a DOM with no CSS engine.
if (!Element.prototype.getAnimations) {
  Element.prototype.getAnimations = () => [];
}

// jsdom has no layout engine, so ResizeObserver (used to follow a growing
// conversation while auto-scrolled to the bottom) doesn't exist either.
// Every real browser has it; this no-op shim just keeps it callable.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class ResizeObserver {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  };
}
