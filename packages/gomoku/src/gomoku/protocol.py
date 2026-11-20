"""Former home of the player interface; see :mod:`gomoku.interfaces`.

``MoveProvider`` is the old name of :class:`gomoku.interfaces.Player` and
stays as an alias so existing imports keep working.
"""

from gomoku.interfaces import Player

MoveProvider = Player

__all__ = ["MoveProvider", "Player"]
