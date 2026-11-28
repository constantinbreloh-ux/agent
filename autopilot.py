"""AutoPILOT-style agent for the Generalist Gaming Agent competition.

step() never waits for the model. The model (`think`) runs in a background thread and is only woken by
events. Its decision is one game action, a controller (Python code it writes, which then plays every
decision point from the screen), a control check, KEEP or REUSE of a stored controller.

Without the model, about once a second, the agent measures image signals: new screen content, changes in
the progress display the model located (unless it ticks like a clock), repeated actions, and wakes the
model when nothing moves for STUCK_S. Screens are told apart: a play screen is one a controller played on
or the agent stayed on for PLAY_S; leaving one ends an attempt (game over or next level). Menu screens
remember the actions that led off them, and the agent replays those when the screen shows again. Every
controller is kept per game in a library with its results; when a play screen shows again its best
controller restarts at once, and model-written variants of a working controller are tried in parallel.
"""
import io
import json
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from gga.protocol import Action

CHECK_S = 10               # while a controller plays, the model looks at it this often
CONTROLLER_MAX_S = 0.1     # a controller call slower than this is switched off
SCENE_DIFF = 0.5           # share of pixels that must differ from ~1 s ago for a scene change
SCENE_COOLDOWN_S = 2
SCENE_MATCH = 12           # mean abs difference of 16x16 gray thumbnails under which two screens are one scene
NOVEL_DIFF = 10            # ... over which a screen counts as new content
DISPLAY_DIFF = 6           # mean abs change in the progress display box that counts as the display changing
STUCK_S = 15               # no progress signal this long: wake the model
PROBE_WAIT = 4             # decision points (200 ms) to watch after each control-check action
REPLAY_GAP_S = 0.6         # between replayed restart actions
VARY_AFTER_S = 20          # a controller running this long gets a variant tried
TRIAL_S = 15               # how long a variant plays before it is compared with its parent
BEST_MIN_S = 5             # a controller must have played this long to be restarted as the best
PLAY_S = 20                # a screen the agent stays on this long is a play screen, not a menu
TICKING = 8                # a display that changed in this many of the last 10 seconds is a clock
XY = {'x', 'y', 'x2', 'y2'}
GRID = 1000                # the model's coordinates: 0-999 over the screen, in actions and controllers alike


def to_pixels(params, width, height):
    """0-999 grid coordinates -> the game's pixels, for an action's params and its sequence steps."""
    for target in [params, *params.get('steps', [])]:
        for key, size in (('x', width), ('y', height), ('x2', width), ('y2', height)):
            if target.get(key) is not None:
                target[key] = min(max(int(target[key]), 0), GRID - 1) * (size - 1) // (GRID - 1)
    return params


def thumb(image):
    return np.asarray(image.convert('L').resize((16, 16)), dtype=np.float32).ravel()


class Agent:
    def __init__(self):
        self.pool = ThreadPoolExecutor(1)
        self.variant_pool = ThreadPoolExecutor(1)   # its calls run beside the model's; vLLM batches them
        self.last_t = float('inf')
        self._reset()

    def think(self, obs, event, memory):
        """Override with the model. Returns one of
        {"tool": ..., "params": {...}}         an action, in pixels
        {"controller": "<code>"}              code defining control(frame, mem) -> action dict or None
        {"probe": [(label, action or None)]}  a control check: each action once, then watch what changes
        {"keep": True}                        the running controller plays on
        {"reuse": <library id>}               restart a stored controller

        frame is the screen scaled to GRID x GRID, RGB numpy, so frame[y, x] is grid point (x, y);
        mem is a dict it keeps. Its actions use grid coordinates, which the agent turns into pixels.
        To say why it hands back, it sets mem['handoff'] = '<reason>' before returning None.
        """
        return {'tool': 'skip', 'params': {}}

    def vary(self, obs, entry, memory):
        """Override with the model: code of a variant of library entry `entry` that should do better, or None."""
        return None

    def _reset(self):
        self.future = self.variant = None   # shortcut: a call still running from the last game delays this one
        self.memory = {}                    # per game, shared with the model
        self.library = self.memory['library'] = []
        self.last_frames = self.memory['last_frames'] = deque(maxlen=8)   # (game time, jpeg), one a second
        self.attempts = self.memory['attempts'] = []                      # durations of ended attempts
        self.controller = self.entry = self.trial = None
        self.probes, self.queue, self.queue_t = [], [], 0
        self.wake = 'new game'
        self.events = []                    # (game time, event) log, for debugging
        self.scenes = deque()
        self.scene_t = -SCENE_COOLDOWN_S
        self.scene_thumbs, self.scene_id, self.scene_actions, self.exits = [], None, [], {}
        self.play_scenes, self.scene_entered_t = set(), 0.0
        self.display_ticks, self.ticking = deque(maxlen=10), False
        self.variants = Counter()           # variant trials: kept / dropped
        self.signal_t, self.seen_thumbs, self.box = -1, np.zeros((0, 256), np.float32), None
        self.novel_t = self.display_t = self.stuck_t = self.attempt_t = 0.0
        self.recent = deque(maxlen=10)      # the model's last actions, as (tool, key) patterns
        self.restart_t, self.restart_delays = None, []

    def step(self, obs, tools):
        try:
            return self._step(obs)
        except Exception as e:   # never raise: the harness plays skip and may start a fresh agent without memory
            self.events.append((getattr(obs, 'elapsed_s', None), f'agent error: {e!r}'))
            return Action('skip')

    def _step(self, obs):
        t = obs.elapsed_s
        if t < self.last_t:   # the clock started again: a new game
            self._reset()
        self.last_t = t
        image = Image.open(io.BytesIO(obs.image)).convert('RGB')
        self._check_scene(t, np.asarray(image), image)
        self._signals(t, image, obs.image)
        if self.controller and obs.last_action.get('ok') is False:   # the game refused its last action
            error = obs.last_action.get('error')
            self._stop(t, 'refused', error, wake=f"the game refused the controller's action: {error}")

        action = None
        if self.future and self.future.done():
            future, self.future = self.future, None
            if self.future_scene != self.scene_id:   # decided for an earlier screen, e.g. Retry once play is back
                self.events.append((t, f'dropped a reply decided on screen #{self.future_scene}'))
            else:
                try:
                    action = self._apply(future.result(), t, image)
                except Exception as e:   # bad model reply: tell the model on its next call
                    self._wake(t, f'your last reply failed: {e!r}')
        if self.variant and self.variant.done():
            future, self.variant = self.variant, None
            try:
                code = future.result()
                if code and action is None and self.entry and not self.trial:
                    action = self._start_trial(code, t, image)
            except Exception as e:
                self.events.append((t, f'variant failed: {e!r}'))
        if action is None and self.trial and t >= self.trial['end']:
            action = self._end_trial(t, image)
        if action is None and self.probes:
            action = self._probe(t, image)
        elif action is None and self.queue and t >= self.queue_t:
            action, self.queue_t = self.queue.pop(0), t + REPLAY_GAP_S
        if action is None and self.controller:
            action = self._run_controller(t, image)
        busy = self.controller or self.probes or self.queue or obs.program_left_s
        if not busy and self.future is None and not self.wake:
            self._wake(t, 'ready')   # nothing is playing: ask for the next move
        if self.wake and self.future is None:
            self.future = self.pool.submit(self.think, obs, self.wake, self.memory)
            self.future_scene, self.wake = self.scene_id, None
        e = self.entry
        if (e and not self.trial and self.variant is None and self.runtime(e, t) >= VARY_AFTER_S
                and t >= e.get('vary_t', 0)):
            e['vary_t'] = t + 30
            self.variant = self.variant_pool.submit(self.vary, obs, e, self.memory)
        return action or Action('skip')   # skip lets running inputs play on

    def _wake(self, t, event):
        self.events.append((t, event))
        self.wake = self.wake or event

    # --- decisions ---

    def _apply(self, decision, t, image):
        if decision.get('keep'):
            return None
        if 'probe' in decision:
            self._stop(t, 'ok', 'replaced by a control check')
            self.probes = [{'label': label, 'action': action and Action(action['tool'], action.get('params') or {})}
                           for label, action in decision['probe']]
            self.queue = []
            self.memory['probes'] = []
            return self._probe(t, image)
        if 'reuse' in decision:
            entry = next((e for e in self.library if e['id'] == decision['reuse'] and 'fn' in e), None)
            if entry is None:
                raise ValueError(f"there is no working controller #{decision['reuse']}")
            self._start(entry, t)
            return self._run_controller(t, image)
        if 'controller' in decision:
            self._start(self._new_entry(decision['controller'], t), t)
            return self._run_controller(t, image)
        self._stop(t, 'ok', 'replaced by an action')
        self.probes, self.queue = [], []
        action = Action(decision['tool'], decision.get('params') or {})
        self.recent.append((action.tool, str(action.params.get('key') or action.params.get('keys') or '')))
        self.scene_actions.append(action)
        return action

    # --- controllers and their library ---

    def _new_entry(self, code, t, parent=None):
        first = code.strip().splitlines()[0] if code.strip() else ''
        entry = {'id': len(self.library) + 1, 'scene': self.scene_id, 'code': code, 'parent': parent,
                 'desc': first.lstrip('#').replace('skill:', '').strip() if first.startswith('#') else first[:60],
                 'runtime': 0.0, 'events': 0, 'signal_t': [], 'readings': [], 'status': 'new', 'reason': ''}
        self.library.append(entry)
        namespace = {'np': np}
        try:
            exec(code, namespace)   # trusted: only our own model writes this code
            entry['fn'] = namespace['control']
        except Exception as e:
            entry['status'], entry['reason'] = 'crashed', repr(e)
            raise
        return entry

    def runtime(self, entry, t):
        return entry['runtime'] + (t - entry['since'] if entry['status'] == 'running' else 0)

    def rate(self, entry, t):
        """Progress signals (new screen content, display changes) per minute while it played."""
        return entry['events'] * 60 / max(self.runtime(entry, t), 1)

    @staticmethod
    def gain(entry):
        """Per minute, how the progress number the model read changed while it played (0 without two readings)."""
        (t0, v0), (t1, v1) = (entry['readings'][0], entry['readings'][-1]) if len(entry['readings']) > 1 else ((0, 0), (1, 0))
        return (v1 - v0) * 60 / max(t1 - t0, 1)

    def _best(self, scene, t):
        good = [e for e in self.library if e['scene'] == scene and e['status'] == 'ok'
                and (e['events'] or len(e['readings']) > 1) and self.runtime(e, t) >= BEST_MIN_S]
        return max(good, key=lambda e: (self.gain(e), self.rate(e, t)), default=None)

    def _start(self, entry, t):
        if self.entry is not entry:
            self._stop(t, 'ok', f"replaced by #{entry['id']}")
        self.controller, self.entry, self.controller_mem = entry['fn'], entry, {}
        self.play_scenes.add(self.scene_id)
        self.controller_start = self.controller_t = entry['since'] = t
        entry['status'], self.probes, self.queue = 'running', [], []

    def _stop(self, t, status, reason, wake=None):
        e = self.entry
        if e:
            e['runtime'] = self.runtime(e, t)
            e['status'], e['reason'] = status, reason
            if status == 'ok' and e.get('stuck'):
                e['status'] = 'no progress'
        self.controller = self.entry = None
        if wake:
            self._wake(t, wake)

    def _run_controller(self, t, image):
        frame = np.asarray(image.resize((GRID, GRID), Image.NEAREST))   # NEAREST: no blended colors at edges
        start = time.time()
        try:
            # shortcut: an endless loop in control() hangs step(); move it to a subprocess if models write those
            choice = self.controller(frame, self.controller_mem)
            if choice is not None:   # numpy -> plain numbers, coordinates -> ints, no empty keys: as the game requires
                choice = json.loads(json.dumps(choice, default=lambda o: o.item()), object_hook=lambda d: {
                    k: round(v) if k in XY and isinstance(v, float) else [x for x in v if x != ''] if k == 'keys'
                    and isinstance(v, list) else v for k, v in d.items()})
                choice['params'] = to_pixels(choice.get('params') or {}, *image.size)
        except Exception as e:
            self._stop(t, 'crashed', repr(e), wake=f'controller crashed: {e!r}')
            return None
        took = time.time() - start
        if took > CONTROLLER_MAX_S:
            self._stop(t, 'too slow', f'{took:.2f} s per call', wake=f'controller too slow ({took:.2f} s per call), stopped')
            return None
        if choice is None:
            reason = self.controller_mem.get('handoff', 'controller handed control back')
            self._stop(t, 'ok', reason, wake=reason)
            return None
        if self.restart_t is not None:   # measures how fast play resumes after a game over
            self.restart_delays.append(round(t - self.restart_t, 1))
            self.restart_t = None
        if t - self.controller_t >= CHECK_S and not self.trial:
            self.controller_t = t   # it keeps playing while the model looks
            self._wake(t, f'check-in: your controller has played for {t - self.controller_start:.0f} s and plays on '
                          'while you look. Is it making progress on what counts (your game note)? '
                          'KEEP it, or reply with a better controller or an action')
        return Action(choice['tool'], choice.get('params') or {})

    def _start_trial(self, code, t, image):
        parent = self.entry
        entry = self._new_entry(code, t, parent=parent['id'])
        self._start(entry, t)
        parent['status'] = 'ok'   # paused, not failed
        self.trial = {'entry': entry, 'parent': parent, 'start': t, 'end': t + TRIAL_S}
        self.events.append((t, f"trying variant #{entry['id']} of #{parent['id']}"))
        return self._run_controller(t, image)

    def _end_trial(self, t, image):
        trial, self.trial = self.trial, None
        variant, parent = trial['entry'], trial['parent']
        if self.entry is not variant:   # it was stopped or replaced meanwhile
            return None
        # the parent's last TRIAL_S before the trial against the variant's TRIAL_S: new screen content gets
        # rarer as a game goes on, so a lifetime average would favour the parent
        start = trial['start']
        ours = sum(s >= start for s in variant['signal_t'])
        theirs = sum(start - TRIAL_S <= s < start for s in parent['signal_t'])
        if ours and ours > 1.2 * theirs:
            self.variants['kept'] += 1
            self.events.append((t, f"variant #{variant['id']} beats #{parent['id']} ({ours} vs {theirs} signals): kept"))
            return None
        self.variants['dropped'] += 1
        self._stop(t, f"worse than #{parent['id']}", f'{ours} vs {theirs} progress signals in {TRIAL_S} s')
        self.events.append((t, f"variant #{variant['id']} is not better than #{parent['id']}: back to #{parent['id']}"))
        self._start(parent, t)
        return self._run_controller(t, image)

    # --- control check ---

    def _probe(self, t, image):
        small = np.asarray(image.convert('L').resize((100, 100), Image.NEAREST), np.float32)
        probe = self.probes[0]
        if 'before' not in probe:
            probe['before'], probe['n'] = small, 0
            return probe['action']   # None: watch the screen without input, for its own motion
        probe['n'] += 1
        if probe['n'] < PROBE_WAIT:
            return None
        changed = np.abs(small - probe['before']) > 30
        if changed.mean() < 0.005:
            result = 'nothing visible changed'
        else:
            ys, xs = np.nonzero(changed)
            result = (f'{changed.mean():.0%} of the screen changed, around x {xs.min() * 10}-{xs.max() * 10 + 9}, '
                      f'y {ys.min() * 10}-{ys.max() * 10 + 9}')
        self.memory['probes'].append(f"{probe['label']}: {result}")
        self.probes.pop(0)
        if not self.probes:
            self._wake(t, 'control check done: see its results')
        return None

    # --- image signals, about once a second ---

    def _signals(self, t, image, jpeg):
        if t - self.signal_t < 1:
            return
        self.signal_t = t
        self.last_frames.append((t, jpeg))
        progress = False
        small = thumb(image)
        if not len(self.seen_thumbs) or np.abs(self.seen_thumbs - small).mean(axis=1).min() > NOVEL_DIFF:
            self.novel_t, progress = t, True
            if len(self.seen_thumbs) < 3000:
                self.seen_thumbs = np.vstack([self.seen_thumbs, small])
        box = self.memory.get('display_box')   # the progress display the model located, grid x0, y0, x1, y1
        if box:
            px = to_pixels({'x': box[0], 'y': box[1], 'x2': box[2], 'y2': box[3]}, *image.size)
            crop = np.asarray(image.convert('L').crop((px['x'], px['y'], px['x2'] + 1, px['y2'] + 1))
                              .resize((32, 8)), np.float32)
            changed = bool(self.box is not None and np.abs(crop - self.box).mean() > DISPLAY_DIFF)
            self.display_ticks.append(changed)
            self.ticking = bool(sum(self.display_ticks) >= TICKING)   # a clock or timer, not progress
            if changed and not self.ticking:
                self.display_t, progress = t, True
            self.box = crop
        if progress and self.entry:
            self.entry['events'] += 1
            self.entry['signal_t'].append(t)
        pattern, count = Counter(self.recent).most_common(1)[0] if self.recent else (None, 0)
        repeating = count >= 8
        self.memory['signals'] = (
            f'new screen content {t - self.novel_t:.0f} s ago; '
            + ('the progress display changes every second like a clock, not counted; ' if box and self.ticking
               else f'progress display changed {t - self.display_t:.0f} s ago; ' if box else 'no progress display located; ')
            + (f'{count} of your last 10 actions were {pattern[0]} {pattern[1]}; ' if repeating else '')
            + f'attempt {len(self.attempts) + 1}, running {t - self.attempt_t:.0f} s'
            + (', earlier attempts lasted ' + ', '.join(f'{a:.0f} s' for a in self.attempts[-5:]) if self.attempts else ''))
        quiet = t - self.novel_t >= STUCK_S and (repeating or not box or self.ticking or t - self.display_t >= STUCK_S)
        if quiet and t - self.stuck_t >= STUCK_S and not self.probes:
            self.stuck_t = t
            if self.entry:
                self.entry['stuck'] = self.entry.get('stuck', 0) + 1
            self._wake(t, f"no progress signal: {self.memory['signals']}. Try something fundamentally different, "
                          'e.g. controls from your note you have not used yet')

    # --- scenes: screen changes, game over, restart ---

    def _scene_of(self, small):
        if self.scene_thumbs:
            dist = np.abs(np.array(self.scene_thumbs) - small).mean(axis=1)
            if dist.min() < SCENE_MATCH:
                return int(dist.argmin()), True
        self.scene_thumbs.append(small)
        return len(self.scene_thumbs) - 1, False

    def _check_scene(self, t, frame, image):
        if self.scene_id is None:
            self.scene_id = self._scene_of(thumb(image))[0]
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
                self._new_scene(t, image)

    def _new_scene(self, t, image):
        old = self.scene_id
        new, known = self._scene_of(thumb(image))
        if new == old:
            return
        played = old in self.play_scenes or t - self.scene_entered_t >= PLAY_S
        if played:
            self.play_scenes.add(old)
        elif self.scene_actions:          # a menu our actions led off: replay them when it shows again
            self.exits[old] = self.scene_actions[-4:]
        self.scene_actions = []
        self._stop(t, 'ok', 'the scene changed')
        self.queue = []                   # a replay that worked stops here, before it clicks into the next screen
        self.scene_id, self.scene_entered_t = new, t
        ended = t - self.attempt_t        # measured before the next attempt's clock starts
        self.attempt_t = t
        if played:                        # a play screen gave way: the attempt ended (game over or next level)
            self.attempts.append(ended)
            self.restart_t = t
            self.queue, self.queue_t = list(self.exits.get(new, [])), t + REPLAY_GAP_S
            self._wake(t, f'game over or next level: play screen #{old} gave way to screen #{new} after {ended:.0f} s'
                          + (f'; replaying the {len(self.queue)} actions that led off this screen last time'
                             if self.queue else '')
                          + '. If the run ended: what ended it, and what do you change?')
            return
        best = self._best(new, t)
        if best:
            self._start(best, t)
        self._wake(t, f"the scene changed (screen #{new}, {'seen before' if known else 'new'})"
                      + (f"; your best controller for it, #{best['id']}, is playing again" if best else ''))
