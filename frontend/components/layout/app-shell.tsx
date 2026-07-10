"use client";

import { useEffect, useState, type ReactNode } from "react";

import { ErrorBoundary } from "@/components/feedback/error-boundary";
import { GlobalSearchDialog } from "@/components/navigation/global-search-dialog";
import { Sidebar } from "@/components/navigation/sidebar";
import { StatusArea } from "@/components/navigation/status-area";
import { TopNav } from "@/components/navigation/top-nav";
import { cn } from "@/lib/utils";
import { useWorkspaceStore } from "@/store/workspace-store";

/**
 * The permanent application frame (Phase 4A). Rendered once by the route
 * group's layout and never destroyed by navigation between Landing,
 * Workspace, and Settings -- only `children` (the routed page) changes.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const sidebarOpen = useWorkspaceStore((state) => state.sidebarOpen);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key === "k") {
        event.preventDefault();
        setSearchOpen((open) => !open);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <div className="flex h-screen flex-col">
      <ErrorBoundary label="Navigation">
        <TopNav onOpenSearch={() => setSearchOpen(true)} />
      </ErrorBoundary>

      <div className="flex min-h-0 flex-1">
        <div
          className={cn(
            "shrink-0 overflow-hidden transition-[width] duration-200",
            sidebarOpen ? "w-64" : "w-0",
          )}
        >
          <ErrorBoundary label="Document list">
            <Sidebar />
          </ErrorBoundary>
        </div>

        <main className="min-w-0 flex-1 overflow-hidden">
          <ErrorBoundary label="Workspace">{children}</ErrorBoundary>
        </main>
      </div>

      <ErrorBoundary label="Status bar">
        <StatusArea />
      </ErrorBoundary>

      <ErrorBoundary label="Search">
        <GlobalSearchDialog open={searchOpen} onOpenChange={setSearchOpen} />
      </ErrorBoundary>
    </div>
  );
}
