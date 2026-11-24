import { useEffect, useRef } from "react";
import { t } from "../lib/i18n";
import { WhiteFlagIcon } from "./icons";
import { Button } from "./ui";

/** "Resign the game?" — Resign (red) / Keep playing.  Keep playing has the focus. */
export function ResignDialog({
  open,
  onResign,
  onClose,
}: {
  open: boolean;
  onResign: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    else if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
      aria-labelledby="resign-title"
      className="m-auto w-[min(100vw-2rem,22rem)] rounded-3xl border border-line bg-surface-strong p-0 text-fg shadow-2xl"
    >
      <div className="flex flex-col gap-5 p-6">
        <div className="flex items-center gap-3">
          <span className="grid size-10 place-items-center rounded-full bg-black/5 dark:bg-white/10">
            <WhiteFlagIcon className="size-5" />
          </span>
          <h2 id="resign-title" className="text-lg font-bold tracking-tight">
            {t.resignTitle}
          </h2>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button variant="danger" onClick={onResign}>
            {t.resignConfirm}
          </Button>
          <Button onClick={onClose} autoFocus>
            {t.keepPlaying}
          </Button>
        </div>
      </div>
    </dialog>
  );
}
