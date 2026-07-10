import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Accessibility preferences (Phase 4C: "Accessibility preferences should
 * override decorative behavior"). `reducedMotion`/`highContrast` are
 * `null` until the user makes an explicit choice, so the system
 * preference (`prefers-reduced-motion`/`prefers-contrast`) is respected
 * by default -- an explicit choice here always wins over it.
 */
interface AccessibilityState {
  reducedMotion: boolean | null;
  highContrast: boolean | null;

  setReducedMotion: (value: boolean | null) => void;
  setHighContrast: (value: boolean | null) => void;
}

export const useAccessibilityStore = create<AccessibilityState>()(
  persist(
    (set) => ({
      reducedMotion: null,
      highContrast: null,

      setReducedMotion: (value) => set({ reducedMotion: value }),
      setHighContrast: (value) => set({ highContrast: value }),
    }),
    { name: "accessibility-store" },
  ),
);
