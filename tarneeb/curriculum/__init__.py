"""Phase 7 curriculum control, history ablations and stage evaluation.

Everything here is additive: it drives the frozen Phase 1-6 engine, environment,
networks, PPO update, checkpointing and evaluation arena without modifying any of
them. No game rule, reward, mask, network or PPO step is reimplemented.
"""
