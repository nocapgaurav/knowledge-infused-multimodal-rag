/**
 * Motion tokens (Phase 3: motion exists only to guide attention, communicate
 * state, and reduce perceived waiting -- never bounce, elastic, or long
 * transitions). Components import these instead of inventing durations or
 * easing curves inline.
 */
export const MOTION_DURATION = {
  fast: 0.12,
  base: 0.2,
  slow: 0.32,
} as const;

export const MOTION_EASING = {
  standard: [0.4, 0, 0.2, 1] as const,
  enter: [0, 0, 0.2, 1] as const,
  exit: [0.4, 0, 1, 1] as const,
};

/** Framer Motion variants for the handful of purposeful transitions this
 * product actually needs: appearing content and panel reveals. */
export const FADE_IN_VARIANTS = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { duration: MOTION_DURATION.base, ease: MOTION_EASING.enter },
  },
};

export const SLIDE_UP_VARIANTS = {
  hidden: { opacity: 0, y: 6 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: MOTION_DURATION.base, ease: MOTION_EASING.enter },
  },
};

/** Respect prefers-reduced-motion: callers should pass this through to
 * Framer Motion's `transition` when the user has requested reduced motion. */
export const REDUCED_MOTION_TRANSITION = { duration: 0 };
