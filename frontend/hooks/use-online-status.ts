"use client";

import { useEffect, useState } from "react";

/**
 * Real browser connectivity, not backend reachability (Phase 4C: "Detect
 * connectivity changes. Communicate offline status clearly."). Distinct
 * from `useBackendHealth` -- a user can be online with the backend down,
 * or (rarer) offline with a cached page still showing a stale "connected"
 * status; this hook only ever reports the former.
 */
export function useOnlineStatus(): boolean {
  const [online, setOnline] = useState(true);

  useEffect(() => {
    setOnline(navigator.onLine);
    function handleOnline() {
      setOnline(true);
    }
    function handleOffline() {
      setOnline(false);
    }
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  return online;
}
