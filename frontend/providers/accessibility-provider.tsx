"use client";

import { useEffect, type ReactNode } from "react";

import { useAccessibilityStore } from "@/store/accessibility-store";

/**
 * Applies accessibility preferences to the document root as data
 * attributes, which `app/globals.css` keys off of. An explicit user
 * choice always overrides the system `prefers-reduced-motion`/
 * `prefers-contrast` media queries, which otherwise remain the default.
 */
export function AccessibilityProvider({ children }: { children: ReactNode }) {
  const reducedMotion = useAccessibilityStore((state) => state.reducedMotion);
  const highContrast = useAccessibilityStore((state) => state.highContrast);

  useEffect(() => {
    const root = document.documentElement;
    if (reducedMotion === null) {
      root.removeAttribute("data-reduced-motion");
    } else {
      root.setAttribute("data-reduced-motion", String(reducedMotion));
    }
  }, [reducedMotion]);

  useEffect(() => {
    const root = document.documentElement;
    if (highContrast === null) {
      root.removeAttribute("data-high-contrast");
    } else {
      root.setAttribute("data-high-contrast", String(highContrast));
    }
  }, [highContrast]);

  return children;
}
