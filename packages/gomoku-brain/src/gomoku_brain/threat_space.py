"""Which threats a threat-space search plays and answers.

A :class:`ThreatSpace` is handed to VCF, VCT and PNS.  Its *attack* fields
choose the moves the attacker may force with; its *defence* fields choose
the replies the defender gets at each forced move.  The threat brain builds
one for its own forced-win search and one for the opponent's (see
``[brains.threat.threats]`` in ``gomoku.toml``).

The defaults are the brain's measured configuration.  Two kinds of change
have different consequences:

* Fewer **attack** threats only make the search less complete: it finds
  fewer wins, never false ones.  More attack threats find more wins but
  widen the search, so a fixed node budget proves less.
* Fewer **defence** replies make proofs *unsound*: the search can "prove"
  wins a reply it did not try refutes, and the brain plays them.  Each
  field's comment says what it guards against.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThreatSpace:
    # -- Attack: threats the attacker may play --------------------------------
    #: Fours (one cell then completes five).  Off: VCF finds nothing and
    #: PNS / VCT attack with threes only.
    fours: bool = True
    #: Split fours (XXX_X, XX_XX) count as fours.  Off: only unbroken runs.
    split_fours: bool = True
    #: Threes (one more stone makes a live four).  Off: PNS / VCT only use fours.
    threes: bool = True
    #: Split threes (_XX_X_) count as threes, defended at the gap or either
    #: end.  Widens the search a lot (-58 Elo at 30k PNS nodes, both sides).
    split_threes: bool = False

    # -- Defence: replies the defender gets ------------------------------------
    #: Block the attacker's four.  Off: every four is treated as winning
    #: (false proofs everywhere).
    block_fours: bool = True
    #: Block the attacker's three (its ends, or gap and ends).  Off: every
    #: three is answered only by counter-threats (false proofs).
    block_threes: bool = True
    #: The defender's own fours as replies.  Off: the "K8" class of false
    #: proofs (an attack refuted by a counter-four) comes back.
    counter_fours: bool = True
    #: Split fours count as the defender's counter-fours.  Off: the F5/E9
    #: class of false proofs (refuted by a split counter-four) comes back.
    split_counter_fours: bool = True
    #: The defender's own threes as replies (PNS).  A heuristic: threat-space
    #: search stays sound without them for attacks made of fours.
    counter_threes: bool = True
    #: Split threes count as counter-threes.
    split_counter_threes: bool = False
    #: Every cell that defuses an attacker's threat to make an unblockable
    #: four next move (PNS).  Off: the #42 class of false proofs comes back.
    unblockable_four_defences: bool = True


#: The brain's default: the configuration measured as strongest.
DEFAULT_THREAT_SPACE = ThreatSpace()

#: Detection as it was before split fours were seen (unbroken runs only).
RUNS_ONLY_THREAT_SPACE = ThreatSpace(split_fours=False, split_counter_fours=False)
