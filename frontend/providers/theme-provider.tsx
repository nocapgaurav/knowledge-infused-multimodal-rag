"use client";

import { ThemeProvider as NextThemeProvider } from "next-themes";
import type { ComponentProps } from "react";

/**
 * Light / dark / system theme, switched instantly with no reload
 * (Phase 4C: "Theme changes should occur instantly. No application reload
 * should be required."). `next-themes` toggles the `.dark` class Tailwind
 * and app/globals.css already key every color token off of.
 */
export function ThemeProvider({ children, ...props }: ComponentProps<typeof NextThemeProvider>) {
  return (
    <NextThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
      {...props}
    >
      {children}
    </NextThemeProvider>
  );
}
