"use client";

import type { ReactNode } from "react";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ErrorBoundary } from "@/components/feedback/error-boundary";

import { AccessibilityProvider } from "@/providers/accessibility-provider";
import { PdfProvider } from "@/providers/pdf-provider";
import { QueryProvider } from "@/providers/query-provider";
import { ThemeProvider } from "@/providers/theme-provider";

/**
 * The single place every global provider is composed, kept shallow
 * (Phase 2A: "Provider nesting should remain shallow. Avoid Provider
 * Hell."). Order matters only where a provider consumes another: Theme
 * must wrap everything so both light/dark tokens are available
 * immediately, and the root ErrorBoundary must be the outermost boundary
 * so a crash anywhere below it still renders the app shell's own chrome.
 */
export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <AccessibilityProvider>
        <QueryProvider>
          <TooltipProvider>
            <PdfProvider>
              <ErrorBoundary label="The application">{children}</ErrorBoundary>
              <Toaster />
            </PdfProvider>
          </TooltipProvider>
        </QueryProvider>
      </AccessibilityProvider>
    </ThemeProvider>
  );
}
