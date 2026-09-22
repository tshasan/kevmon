"""poke-engine bridge: turning a live poke-env Battle into a searchable poke_engine.State.

The engine is a separate rules implementation from Showdown's, compiled for one generation at a time, and it
validates almost nothing at the Python boundary -- so this package is deliberately thin, explicit and tested
against the real engine rather than against its type stubs (which are stale).

    vocab.py   poke-env enum names -> poke-engine enum strings; the crossing that panics if you get it wrong
    stats.py   final stats for opponents, following gen9 randbats' own EV/IV rules
"""
