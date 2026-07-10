"use client";

import { ThemeToggle } from "@/components/settings/theme-toggle";
import { Switch } from "@/components/ui/switch";
import { TYPOGRAPHY } from "@/constants/typography";
import { config } from "@/lib/config";
import { cn } from "@/lib/utils";
import { useBackendHealth } from "@/services/queries";
import { useAccessibilityStore } from "@/store/accessibility-store";
import { useWorkspaceStore } from "@/store/workspace-store";

/**
 * Settings stay intentionally minimal (Phase 4C: "Avoid configuration
 * overload") -- every control here has a real, immediate effect; nothing
 * is a placeholder section reserved for a future feature.
 */
export function SettingsPanel() {
  const { data, isError, isPending } = useBackendHealth();
  const connected = !isPending && !isError && data?.status === "ok";

  const sidebarOpen = useWorkspaceStore((state) => state.sidebarOpen);
  const toggleSidebar = useWorkspaceStore((state) => state.toggleSidebar);

  const reducedMotion = useAccessibilityStore((state) => state.reducedMotion);
  const setReducedMotion = useAccessibilityStore((state) => state.setReducedMotion);
  const highContrast = useAccessibilityStore((state) => state.highContrast);
  const setHighContrast = useAccessibilityStore((state) => state.setHighContrast);

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-8">
      <h1 className={TYPOGRAPHY.panelTitle}>Settings</h1>

      <SettingsSection title="Appearance">
        <div className="flex items-center justify-between">
          <span className={TYPOGRAPHY.body}>Theme</span>
          <ThemeToggle />
        </div>
      </SettingsSection>

      <SettingsSection title="Workspace">
        <SettingRow
          label="Show document sidebar"
          checked={sidebarOpen}
          onCheckedChange={toggleSidebar}
        />
      </SettingsSection>

      <SettingsSection title="Accessibility">
        <SettingRow
          label="Reduce motion"
          checked={reducedMotion ?? false}
          onCheckedChange={setReducedMotion}
        />
        <SettingRow
          label="High contrast"
          checked={highContrast ?? false}
          onCheckedChange={setHighContrast}
        />
      </SettingsSection>

      <SettingsSection title="Connection">
        <div className="flex items-center justify-between">
          <span className={TYPOGRAPHY.body}>Backend</span>
          <span className={cn(TYPOGRAPHY.caption, connected ? "text-success" : "text-error")}>
            {isPending ? "Checking..." : connected ? "Connected" : "Unavailable"}
          </span>
        </div>
      </SettingsSection>

      <SettingsSection title="About">
        <dl className="flex flex-col gap-1">
          <Row label="Application" value={config.app.name} />
          <Row label="Version" value={config.app.version} />
        </dl>
      </SettingsSection>
    </div>
  );
}

function SettingsSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className={TYPOGRAPHY.sectionTitle}>{title}</h2>
      {children}
    </section>
  );
}

function SettingRow({
  label,
  checked,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <span id={`setting-${label}`} className={TYPOGRAPHY.body}>
        {label}
      </span>
      <Switch
        checked={checked}
        onCheckedChange={onCheckedChange}
        aria-labelledby={`setting-${label}`}
      />
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <dt className={TYPOGRAPHY.caption}>{label}</dt>
      <dd className={TYPOGRAPHY.body}>{value}</dd>
    </div>
  );
}
