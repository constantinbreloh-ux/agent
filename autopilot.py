"""AutoPILOT-style agent for the Generalist Gaming Agent competition.

step() never waits for the model. The model (`think`) runs in a background thread and is only
woken by events. Its decision is either one game action or a controller: Python code the model
writes, which then plays every decision point (200 ms) from the screen until it hands back.
"""
import io
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from gga.protocol import Action

TIMEOUT_S = 120            # wake the model if a controller ran this long without it
CONTROLLER_MAX_S = 0.1     # a controller call slower than this is switched off
SCENE_DIFF = 0.5           # share of pixels that must differ from ~1 s ago for a scene change
SCENE_COOLDOWN_S = 2


class Agent:
    def __init__(self):
        self.pool = ThreadPoolExecutor(1)
        self.last_t = float('inf')
        self._reset()

    def think(self, obs, event, memory):
        """Override with the model. Returns {"tool": ..., "params": {...}} or {"controller": "<code>"}.

        Controller code defines control(frame, mem) -> action dict, or None to hand control back.
        frame is the screen as an RGB numpy array (pixels = action coordinates), mem a dict it keeps.
        To say why it hands back, it sets mem['handoff'] = '<reason>' before returning None.
        """
        return {'tool': 'skip', 'params': {}}

    def _reset(self):
        self.future = None   # shortcut: a think still running from the last game delays this game's first one
        self.memory = {}     # per game, for the model's notes
        self.controller = None
        self.busy_until = 0  # game time until which a sequence is still playing
        self.wake = 'new game'
        self.scenes = deque()
        self.scene_t = -SCENE_COOLDOWN_S
        self.events = []     # (game time, event) log, for debugging

    def step(self, obs, tools):
        t = obs.elapsed_s
        if t < self.last_t:   # the clock started again: a new game
            self._reset()
        self.last_t = t
        frame = np.asarray(Image.open(io.BytesIO(obs.image)).convert('RGB'))
        self._check_scene(t, frame)

        action = None
        if self.future and self.future.done():
            future, self.future = self.future, None
            try:
                action = self._apply(future.result(), t, frame)
            except Exception as e:   # bad model reply: tell the model on its next call
                self._wake(t, f'your last reply failed: {e!r}')
        if action is None and self.controller:
            action = self._run_controller(t, frame)
        if self.controller is None and self.future is None and t >= self.busy_until and not self.wake:
            self._wake(t, 'ready')   # nothing is playing: ask for the next move
        if self.wake and self.future is None:
            self.future = self.pool.submit(self.think, obs, self.wake, self.memory)
            self.wake = None
        return action or Action('skip')   # skip lets running inputs play on

    def _wake(self, t, event):
        self.events.append((t, event))
        self.wake = self.wake or event

    def _apply(self, decision, t, frame):
        if 'controller' in decision:
            namespace = {'np': np}
            exec(decision['controller'], namespace)   # trusted: only our own model writes this code
            self.controller, self.controller_mem, self.controller_t = namespace['control'], {}, t
            return self._run_controller(t, frame)
        self.controller = None
        steps = decision.get('params', {}).get('steps', []) if decision['tool'] == 'sequence' else []
        self.busy_until = t + sum(s.get('duration_ms', 0) for s in steps) / 1000
        return Action(decision['tool'], decision.get('params', {}))

    def _run_controller(self, t, frame):
        start = time.time()
        try:
            # shortcut: an endless loop in control() hangs step(); move it to a subprocess if models write those
            choice = self.controller(frame, self.controller_mem)
            if choice is not None:   # numpy numbers -> plain ints/floats, which the game requires
                choice = json.loads(json.dumps(choice, default=lambda o: o.item()))
        except Exception as e:
            self.controller = None
            self._wake(t, f'controller crashed: {e!r}')
            return None
        if time.time() - start > CONTROLLER_MAX_S:
            self.controller = None
            self._wake(t, f'controller too slow ({time.time() - start:.2f} s per call), stopped')
        elif choice is None:
            self.controller = None
            self._wake(t, self.controller_mem.get('handoff', 'controller handed control back'))
            return None
        elif t - self.controller_t > TIMEOUT_S:
            self.controller_t = t
            self._wake(t, 'controller still running after 120 s')   # it keeps playing while the model looks
        return Action(choice['tool'], choice.get('params', {}))

    def _check_scene(self, t, frame):
        small = frame[::8, ::8].mean(axis=2)
        self.scenes.append((t, small))
        while self.scenes[0][0] < t - 1:
            old = self.scenes.popleft()[1]
            if t - self.scene_t < SCENE_COOLDOWN_S or len(self.scenes) < 2:
                continue
            changed = (np.abs(small - old) > 24).mean()
            holds = (np.abs(small - self.scenes[-2][1]) > 24).mean() < 0.05
            if changed > SCENE_DIFF and holds:
                self.scene_t = t
                self.controller = None
                self._wake(t, 'the scene changed')
