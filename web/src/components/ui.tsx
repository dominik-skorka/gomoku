import { clsx } from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost";

export function Button({
  variant = "secondary",
  icon = false,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; icon?: boolean }) {
  return (
    <button
      type="button"
      className={clsx(
        "inline-flex h-10 items-center justify-center gap-2 rounded-xl text-sm font-semibold transition",
        icon ? "w-10" : "px-4",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-40 active:scale-[0.97]",
        variant === "primary" && "bg-accent text-accent-fg shadow-sm hover:brightness-110",
        variant === "secondary" && "border border-line bg-surface-strong hover:bg-surface",
        variant === "ghost" && "hover:bg-surface",
        className,
      )}
      {...props}
    />
  );
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("rounded-2xl border border-line bg-surface p-4 shadow-sm backdrop-blur-xl", className)}>
      {children}
    </div>
  );
}

export function StoneIcon({ stone, className }: { stone: 1 | 2; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={clsx(
        "inline-block size-4 shrink-0 rounded-full shadow",
        stone === 1
          ? "bg-[radial-gradient(circle_at_35%_30%,#6b7180,#1d2028_45%,#050608)]"
          : "bg-[radial-gradient(circle_at_35%_30%,#fff,#e9ebef_70%,#b9bec8)] ring-1 ring-black/10",
        className,
      )}
    />
  );
}
