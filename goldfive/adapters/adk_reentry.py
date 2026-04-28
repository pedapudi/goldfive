"""Re-entry contract for ADK plugins observing a goldfive-wrapped tree.

When goldfive's overlay calls into ADK's Runner with the operator's
verbatim input (or a goldfive-composed nudge/steer message), the
``on_user_message_callback`` (and similar plugin hooks) fire against
the inner runner — but the outer runner already fired them for the
operator's actual input. Plugins observing both sides will see
duplicates unless they can distinguish "user turn" from "framework
re-entry."

Goldfive pins :data:`current_reentry_kind` to a non-default value
when re-entering. Plugins read the var and short-circuit duplicate
side-effect emissions. Default is :attr:`ReentryKind.USER_TURN` so
plain ADK use is unchanged.

Background — harmonograf#234 root cause
---------------------------------------

gRPC ground truth for a 4-turn session showed 6 ``UserMessageReceived``
envelopes: turns that completed emitted twice, turns that aborted
mid-invocation emitted once. The duplicate fire originates from the
goldfive overlay re-entering ADK via :meth:`ADKAdapter.invoke_passthrough`
which calls ``self._runner.run_async(new_message=<operator input>)``.
ADK's ``_handle_new_message`` then triggers the plugin manager's
``on_user_message_callback`` a second time, against the same operator
input that was already observed and emitted by the outer adk-web
runner. The harmonograf-client plugin's existing
``_maybe_disable_as_duplicate`` guard only catches multiple plugin
*instances* on the same plugin manager — here we have ONE plugin
instance fired twice in two distinct invocation contexts.

Contract
--------

Goldfive promises:

* The default value (``USER_TURN``) is what every plain-ADK code path
  observes — there is no behaviour change for non-goldfive callers.
* When goldfive re-enters ADK with a message that did NOT originate
  from a fresh operator turn, ``current_reentry_kind`` is set to a
  non-default :class:`ReentryKind` for the duration of the inner
  ``run_async`` call (and any callbacks dispatched from it).
* The contextvar is reset on exit, including on exception.

Plugins observing this contract may suppress duplicate side effects
(e.g. dedup user-message envelope emission). Plugins that only care
about agent-level events (before/after_agent_callback) generally do
NOT need to consult the contextvar — those events are not duplicated
because the outer runner has no inner agents of its own.
"""
from __future__ import annotations

import contextlib
import enum
from collections.abc import Iterator
from contextvars import ContextVar


class ReentryKind(enum.Enum):
    """Discriminator for ``on_user_message_callback`` re-entry shape."""

    #: A fresh operator turn — the message is new user input. Default.
    USER_TURN = "user_turn"

    #: Goldfive overlay re-feeding the operator's verbatim user input
    #: through the inner ADK runner so the wrapped tree sees it in its
    #: session history. Triggered by :meth:`ADKAdapter.invoke_passthrough`.
    OVERLAY_REPLAY = "overlay_replay"

    #: Goldfive-composed nudge replay: the steerer queued a
    #: ``session.pending_nudges`` entry (e.g. after a LOOPING_REASONING
    #: drift caused refine to spawn a replacement task) and the executor
    #: re-invokes passthrough with the composed nudge as the new user
    #: message. Content is goldfive-authored, not operator-typed.
    NUDGE_REPLAY = "nudge_replay"

    #: Goldfive-composed steer-restart replay: the operator issued a
    #: STEER mid-invocation, the executor cancelled the in-flight
    #: invocation, fed the steer through the planner refine, and then
    #: re-invokes passthrough with the steer body wrapped in a
    #: ``USER STEERING CONTROL`` override header. Header-wrapped content
    #: is goldfive-authored framing around operator text.
    STEER_REPLAY = "steer_replay"


current_reentry_kind: ContextVar[ReentryKind] = ContextVar(
    "goldfive_reentry_kind", default=ReentryKind.USER_TURN
)


@contextlib.contextmanager
def reentry(kind: ReentryKind) -> Iterator[ReentryKind]:
    """Pin :data:`current_reentry_kind` to ``kind`` for the duration of the block.

    Stack-precedence semantics
    --------------------------

    A more-specific kind nested inside a less-specific one keeps the
    more-specific value visible. Concretely: a STEER_REPLAY *or*
    NUDGE_REPLAY layered inside an OVERLAY_REPLAY (the natural shape
    when the executor composes a steer/nudge message and then calls
    :meth:`ADKAdapter.invoke_passthrough`, which itself pins
    OVERLAY_REPLAY) keeps the steer/nudge label visible to plugins.
    Plugins thus see the *cause* of the replay, not the mechanism.

    The reverse — entering OVERLAY_REPLAY while already inside a
    STEER_REPLAY or NUDGE_REPLAY — does NOT downgrade the visible
    kind. It yields the prior (more-specific) kind unchanged.

    Yielding from a USER_TURN context ALWAYS pins ``kind`` regardless
    of value, so the first re-entry boundary always wins from the
    default state.

    The contextvar is reset on exit, including on exception, via the
    standard ``ContextVar.reset(token)`` protocol.
    """
    prior = current_reentry_kind.get()
    if prior is ReentryKind.USER_TURN or kind in (
        ReentryKind.STEER_REPLAY,
        ReentryKind.NUDGE_REPLAY,
    ):
        token = current_reentry_kind.set(kind)
        try:
            yield kind
        finally:
            current_reentry_kind.reset(token)
    else:
        # Already in a more-specific re-entry; don't downgrade.
        yield prior
