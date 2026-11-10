"""Isaac Lab side of the environments: FORGE tasks, robot profiles, USD assets and the env server.

Everything in this package imports Isaac Lab / Isaac Sim / torch and runs inside the Isaac Lab virtualenv ONLY, as
`python -m src.envs.isaaclab.server` from the repo root (never `python src/envs/isaaclab/server.py`: its directory would then
shadow the real `isaaclab` package). This __init__ imports nothing, so importing the package costs nothing; the learner never
imports it. The simulator-independent logic (protocol, episode manager and tracker, WebSocket server, client, collector) is in
src/envs/.
"""
