/**
 * Layout tokens: panel sizing and responsive breakpoints (Phase 2A design
 * tokens; Phase 4A/4C workspace + responsive rules). Kept centralized so no
 * component invents its own panel width or breakpoint.
 */
export const PANEL_WIDTH = {
  sidebarDefault: 280,
  sidebarMin: 220,
  sidebarMax: 360,
  conversationDefaultPercent: 50,
  panelMinPercent: 18,
} as const;

/** Mirrors Tailwind's default breakpoints -- documented here for JS-side
 * responsive logic (e.g. matchMedia) so the two never drift apart. */
export const BREAKPOINT_PX = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  "2xl": 1536,
} as const;

export type Breakpoint = keyof typeof BREAKPOINT_PX;
