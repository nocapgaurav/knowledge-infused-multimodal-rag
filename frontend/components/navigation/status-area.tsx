"use client";

import { useEffect, useRef } from "react";
import { Circle } from "lucide-react";
import { toast } from "sonner";

import { TYPOGRAPHY } from "@/constants/typography";
import { useOnlineStatus } from "@/hooks/use-online-status";
import { cn } from "@/lib/utils";
import { useBackendHealth } from "@/services/queries";

/** A lightweight status area (Phase 4A/4C) -- communicates connectivity
 * without ever cluttering the primary workspace. Offline (no network at
 * all) takes precedence over backend-unavailable (network present, but
 * the backend specifically isn't reachable). */
export function StatusArea() {
  const online = useOnlineStatus();
  const { data, isError, isPending } = useBackendHealth();
  const connected = !isPending && !isError && data?.status === "ok";
  const wasOnline = useRef(online);

  useEffect(() => {
    if (wasOnline.current && !online) {
      toast.warning("You're offline", {
        description: "Some actions won't work until you reconnect.",
      });
    } else if (!wasOnline.current && online) {
      toast.success("Back online");
    }
    wasOnline.current = online;
  }, [online]);

  const label = !online
    ? "Offline"
    : isPending
      ? "Checking backend connection..."
      : connected
        ? "Backend connected"
        : "Backend unavailable";

  return (
    <footer className="flex h-7 shrink-0 items-center gap-2 border-t px-4">
      <Circle
        className={cn(
          "size-2 fill-current",
          !online || (!isPending && !connected)
            ? "text-error"
            : isPending
              ? "text-muted-foreground"
              : "text-success",
        )}
      />
      <span className={TYPOGRAPHY.metadata}>{label}</span>
    </footer>
  );
}
