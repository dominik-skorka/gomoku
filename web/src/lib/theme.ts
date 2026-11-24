import { useCallback, useEffect, useState } from "react";

export type ThemeChoice = "system" | "light" | "dark";
const KEY = "gomoku.theme";

function read(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

/** Theme preference: `system` follows `prefers-color-scheme`; the choice is remembered per browser. */
export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(read);
  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") delete root.dataset.theme;
    else root.dataset.theme = choice;
    try {
      if (choice === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, choice);
    } catch {
      // Storage blocked (private mode): the choice lasts for this page only.
    }
  }, [choice]);
  const cycle = useCallback(() => setChoice((c) => (c === "system" ? "light" : c === "light" ? "dark" : "system")), []);
  return { choice, cycle };
}
