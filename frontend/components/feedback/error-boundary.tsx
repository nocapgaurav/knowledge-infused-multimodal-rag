"use client";

import { Component, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { TYPOGRAPHY } from "@/constants/typography";

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Shown in the fallback so users understand *what* failed without ever
   * seeing a stack trace (Phase 4C: "Errors should explain the problem...
   * Never expose stack traces"). */
  label: string;
  /** Optional fully custom fallback; the default one already satisfies
   * the error-experience rules, so most callers won't need this. */
  fallback?: (retry: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

/**
 * A reusable, per-panel error boundary (Phase 4D: "Panels should remain
 * independent. Failure inside one panel must never crash another.").
 * React error boundaries must be class components -- there is no hook
 * equivalent -- so this is the one class component in the codebase,
 * deliberately isolated here as a Foundation component the rest of the
 * app composes rather than reimplements.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    // Development-only diagnostic; never surfaced to end users (Phase 2B:
    // "Never expose technical logs to end users").
    if (process.env.NODE_ENV !== "production") {
      console.error(`[${this.props.label}] rendering error:`, error);
    }
  }

  private retry = () => this.setState({ hasError: false });

  render() {
    if (!this.state.hasError) return this.props.children;
    if (this.props.fallback) return this.props.fallback(this.retry);

    return (
      <div className="flex h-full min-h-32 flex-col items-center justify-center gap-3 p-6 text-center">
        <AlertTriangle className="text-warning size-6" aria-hidden="true" />
        <p className={TYPOGRAPHY.body}>
          {this.props.label} ran into a problem and can&apos;t be shown right now.
        </p>
        <Button variant="outline" size="sm" onClick={this.retry}>
          <RotateCcw className="size-4" />
          Try again
        </Button>
      </div>
    );
  }
}
