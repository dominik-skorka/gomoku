import { useEffect, useId, useRef, useState } from "react";

/**
 * An (i) button with an explanation in a bubble: shown on hover and focus
 * (mouse, keyboard), toggled by a tap (touch has no hover).  The bubble is
 * `position: fixed`, so a scrolling dialog does not clip it, and it is kept
 * inside the viewport.
 */
export function InfoTip({ text, label }: { text: string; label: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean } | null>(null);
  const button = useRef<HTMLButtonElement>(null);
  const pointer = useRef<string>("");
  const id = useId();

  useEffect(() => {
    if (!open || !button.current) return;
    const r = button.current.getBoundingClientRect();
    const width = Math.min(288, window.innerWidth - 16);
    const left = Math.min(Math.max(8, r.left + r.width / 2 - width / 2), window.innerWidth - width - 8);
    const above = r.top > window.innerHeight / 2;
    setPos({ left, top: above ? r.top - 6 : r.bottom + 6, above });
    const close = (e: Event) => {
      if (e instanceof PointerEvent && button.current?.contains(e.target as Node)) return;
      setOpen(false);
    };
    // Escape closes the bubble only, not the dialog around it.
    const onEscape = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      setOpen(false);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", onEscape, true);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", onEscape, true);
      window.removeEventListener("scroll", close, true);
    };
  }, [open]);

  return (
    <>
      <button
        ref={button}
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        className="grid size-4 shrink-0 place-items-center rounded-full border border-current text-[10px] font-bold italic leading-none text-muted hover:text-fg focus-visible:outline-2 focus-visible:outline-accent"
        // Hover: mouse only (a tap also fires compatibility mouse events).
        onPointerEnter={(e) => e.pointerType === "mouse" && setOpen(true)}
        onPointerLeave={(e) => e.pointerType === "mouse" && setOpen(false)}
        onPointerDown={(e) => {
          pointer.current = e.pointerType;
        }}
        // Focus opens it for keyboard users only; a tap focuses too.
        onFocus={(e) => e.currentTarget.matches(":focus-visible") && setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.preventDefault();
          // A tap toggles; a mouse click keeps the hover bubble open.
          if (pointer.current === "mouse") setOpen(true);
          else setOpen((o) => !o);
        }}
      >
        i
      </button>
      {open && pos && (
        <span
          id={id}
          role="tooltip"
          className="fixed z-50 w-[min(18rem,calc(100vw-1rem))] rounded-lg border border-line bg-fg px-3 py-2 text-xs leading-snug font-normal text-bg shadow-xl"
          style={{ left: pos.left, top: pos.top, transform: pos.above ? "translateY(-100%)" : undefined }}
        >
          {text}
        </span>
      )}
    </>
  );
}
