"""AutoPILOT-style agent for the Generalist Gaming Agent competition: a Python conductor and model roles.

step() never waits for a model. It is the conductor: the only code that changes memory and the controller
library. Model roles run in threads and only return results, each tagged with the screen it was meant for:
- player   (one at a time, fast): an action, KEEP, REUSE #id, or WRITE: <goal> for the writer
- writer   (background): a controller for a goal; it starts by itself if its screen still shows,
           otherwise it goes into the library
- variant  (background): a variant of a working controller, tried for TRIAL_S against its parent
- checker  (background, every CHECK_S while a controller plays): KEEP or REPLACE, with the number it reads;
           REPLACE sends the writer a goal
- analyst  (background): the game note and its revisions, the control check plan, the lesson after a game over
At most MAX_BACKGROUND background calls run at once; the player runs beside them.

Without a model, about once a second, the conductor measures image signals: new screen content against
the last NOVEL_WINDOW_S, changes in the progress display the analyst located (unless it ticks like a
clock), repeated actions. Nothing for STUCK_S makes the checker look at a running controller, or else
wakes the player. A decision counter rises with every change of situation (a controller starts, the scene
changes, a control check or a replay starts); a player reply from an older count is dropped. A play screen is one a controller played
on or the agent stayed on for PLAY_S; leaving one ends an attempt. Menus remember the actions that led off
them and the conductor replays them; a play screen that shows again restarts its best controller.
"""
import io
import json
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from gga.protocol import Action

CHECK_S = 10               # while a controller plays, the checker looks at it this often
CONTROLLER_MAX_S = 0.1     # a controller call slower than this is switched off
SCENE_DIFF = 0.5           # share of pixels that must differ from ~1 s ago for a scene change
SCENE_COOLDOWN_S = 2
SCENE_MATCH = 12           # mean abs difference of 16x16 gray thumbnails under which two screens are one scene
NOVEL_SIZE = 64            # new screen content: 64x64 gray previews, measured on the 8 videos of runs 10 and 11
NOVEL_PIXEL = 20           # a preview pixel changed if its gray value moved more than this
NOVEL_SHARE = 0.005        # new if at least this share of pixels differs from every screen of the window
NOVEL_WINDOW_S = 30        # new against the screens of the last this many seconds, not the whole game
DISPLAY_DIFF = 6           # mean abs change in the progress display box that counts as the display changing
STUCK_S = 15               # no progress signal this long: wake the player
PROBE_WAIT = 4             # decision points (200 ms) to watch after each control-check action
REPLAY_GAP_S = 0.6         # between replayed restart actions
VARY_AFTER_S = 20          # a controller running this long gets a variant tried
TRIAL_S = 15               # how long a variant plays before it is compared with its parent
BEST_MIN_S = 5             # a controller must have played this long to be restarted as the best
PLAY_S = 20                # a screen the agent stays on this long is a play screen, not a menu
TICKING = 8                # a display that changed in this many of the last 10 seconds is a clock
NOTE_RECHECK_S = 20        # the analyst revises the note once at this game time
MAX_BACKGROUND = 2         # background model calls at once
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
        self.pool = ThreadPoolExecutor(1)                        # the player
        self.background = ThreadPoolExecutor(MAX_BACKGROUND)     # writer, variant, checker, analyst
        self.last_t = float('inf')
        self._reset()

    # --- the model roles: override them. Each gets a snapshot `view` (see _view) and returns a result. ---

    def think(self, obs, event, view):
        """The player. Returns a decision:
        {"tool": ..., "params": {...}}         an action, in pixels
        {"keep": True}                        the running controller plays on
        {"reuse": <library id>}               restart a stored controller
        {"write": "<goal>"}                   ask the writer for a controller; keep playing meanwhile
        {"controller": "<code>"} or {"probe": [(label, action or None)]} are accepted too.
        Optional "updates": {"progress": (text, number or None), "note_fix": str, "log": str}.
        """
        return {'tool': 'skip', 'params': {}}

    def write(self, obs, goal, view):
        """The writer: code defining control(frame, mem) -> action dict (grid coordinates) or None to hand
        back, for `goal`. frame is the screen scaled to GRID x GRID, RGB numpy. Or None."""
        return None

    def vary(self, obs, entry, view):
        """The variant writer: code of a variant of `entry` (id, desc, code) that should do better, or None."""
        return None

    def check(self, obs, view):
        """The checker: {"verdict": "KEEP" | "REPLACE", "reason": str, "reading": number or None}, or None.
        view['frames'] holds the last seconds as jpegs, view['playing'] the running controller."""
        return None

    def analyse(self, obs, task, view):
        """The analyst. task 'note' -> {"note": str, "display_box": (x0, y0, x1, y1) or None};
        'probe' -> {"probe": [(label, action or None)]}; 'lesson' -> {"lesson": str}. Or None."""
        return None

    # --- state ---

    def _reset(self):
        self.future = None                  # shortcut: calls still running from the last game delay this one
        self.jobs = []                      # background calls: {key, role, scene, future, ...}
        self.memory = {'note': None, 'fixes': [], 'probes': [], 'lessons': [], 'progress': [], 'log': [],
                       'display_box': None, 'signals': 'nothing yet'}
        self.library = []
        self.last_frames = deque(maxlen=12)   # (game time, jpeg), one a second
        self.attempts = []                    # durations of ended attempts
        self.controller = self.entry = self.trial = None
        self.probes, self.queue, self.queue_t = [], [], 0
        self.wake = 'new game'
        self.writing = None                   # the writer's current goal
        self.notes_due, self.notes_done, self.probed = [], set(), False
        self.events = []                      # (game time, event) log, for debugging
        self.scenes = deque()
        self.scene_t = -SCENE_COOLDOWN_S
        self.scene_thumbs, self.scene_id, self.scene_actions, self.exits = [], None, [], {}
        self.play_scenes, self.scene_entered_t = set(), 0.0
        self.display_ticks, self.ticking = deque(maxlen=10), False
        self.signal_t, self.recent_thumbs, self.box = -1, deque(), None
        self.novel_t = self.display_t = self.stuck_t = self.attempt_t = 0.0
        self.recent = deque(maxlen=10)        # the player's last actions, as (tool, key) patterns
        self.restart_t, self.restart_delays = None, []
        self.variants, self.verdicts, self.stats = Counter(), Counter(), Counter()
        self.first_scene_t = self.first_play_delay = None
        self.epoch = 0                        # the decision counter

    def _view(self, t, **extra):
        """A snapshot for a model role: copies, so threads never see memory change under them."""
        m = self.memory
        e = self.entry
        return {'note': m['note'], 'fixes': list(m['fixes']), 'probes': list(m['probes']),
                'lessons': list(m['lessons']), 'progress': list(m['progress']), 'log': list(m['log']),
                'signals': m['signals'], 'library': self.library_text(t), 'writing': self.writing,
                'playing': e and {'id': e['id'], 'desc': e['desc'], 'code': e['code'],
                                  'seconds': t - self.controller_start, 'readings': list(e['readings'])},
                'frames': [jpeg for _, jpeg in self.last_frames], 'attempts': list(self.attempts),
                'scene': self.scene_id, **extra}

    def library_text(self, t):
        lines = []
        for e in self.library:
            if e['status'] in ('ok', 'running', 'stored') or e['status'].startswith('worse'):
                read = e['readings']
                lines.append(f"#{e['id']} [screen {e['scene']}] {e['desc']}: played {self.runtime(e, t):.0f} s, "
                             + (f"progress read {read[0][1]:g} -> {read[-1][1]:g}, " if len(read) > 1 else '')
                             + f"{self.rate(e, t):.0f} progress signals/min, {e['status']}")
            else:
                lines.append(f"#{e['id']} {e['desc']}: FAILED ({e['status']}: {e['reason'][:100]})")
        return '\n'.join(lines[-12:]) or 'empty'

    # --- the conductor ---

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
        self.last_t, self._obs = t, obs   # _obs: for jobs started deep inside the conductor
        image = Image.open(io.BytesIO(obs.image)).convert('RGB')
        self._check_scene(t, np.asarray(image), image)
        self._signals(t, image, obs.image)
        if self.controller and obs.last_action.get('ok') is False:   # the game refused its last action
            error = obs.last_action.get('error')
            self._stop(t, 'refused', error, wake=f"the game refused the controller's action: {error}")

        action = None
        if self.future and self.future.done():
            future, self.future = self.future, None
            try:
                decision = future.result()
                self._update(decision.pop('updates', None) or {}, t)
                self.stats['player replies'] += 1
                if self.future_epoch != self.epoch:   # decided before the situation changed, e.g. a controller started
                    self.stats['player replies dropped'] += 1
                    self.events.append((t, f'dropped a player reply from decision #{self.future_epoch} (now #{self.epoch})'))
                else:
                    action = self._apply(decision, t, image)
            except Exception as e:   # bad model reply: tell the player on its next call
                self._wake(t, f'your last reply failed: {e!r}')
        self._collect(obs, t)
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
            self._wake(t, 'ready')   # nothing is playing: ask the player for the next move
        if self.wake and self.future is None:
            self.future = self.pool.submit(self.think, obs, self.wake, self._view(t))
            self.future_epoch, self.wake = self.epoch, None
        self._schedule(obs, t)
        self._measure(t, action, obs)
        return action or Action('skip')   # skip lets running inputs play on

    def _wake(self, t, event):
        self.events.append((t, event))
        self.wake = self.wake or event

    def _bump(self):
        self.epoch += 1

    def _measure(self, t, action, obs):
        self.stats['steps'] += 1
        if action is None and not self.controller and not obs.program_left_s:   # nothing plays: idle
            self.stats['idle steps'] += 1
        if any(j['key'] == 'writer' for j in self.jobs):   # how much play goes on while the writer works
            self.stats['steps while writing'] += 1
            self.stats['skips while writing'] += action is None
        if self.first_scene_t is not None and self.first_play_delay is None and action is not None:
            self.first_play_delay = round(t - self.first_scene_t, 1)

    def _update(self, updates, t):
        """Results a role reported about the game, merged here in the conductor."""
        m = self.memory
        if updates.get('log'):
            m['log'] = (m['log'] + [updates['log']])[-20:]
        if updates.get('progress'):
            text, number = updates['progress']
            m['progress'].append((t, text))
            if number is not None and self.entry:   # the number read, to compare controllers by
                self.entry['readings'].append((t, number))
        if updates.get('note_fix'):
            m['fixes'].append(updates['note_fix'])
            if updates.get('display_box'):
                m['display_box'], self.box = updates['display_box'], None

    # --- background jobs ---

    def _job(self, key, role, fn, obs, *args, **info):
        """Start a background call unless one with this key still runs. The pool caps how many run at once."""
        if any(j['key'] == key for j in self.jobs):
            return False
        t = obs.elapsed_s
        future = self.background.submit(fn, obs, *args, self._view(t))
        self.jobs.append({'key': key, 'role': role, 'scene': self.scene_id, 'future': future, 't': t, **info})
        return True

    def _schedule(self, obs, t):
        m = self.memory
        if m['note'] is None and 'start' not in self.notes_done:
            self.notes_done.add('start')
            self._job('analyst:note', 'analyst', self.analyse, obs, 'note', why='start')
        if t >= NOTE_RECHECK_S and 'time' not in self.notes_done:
            self.notes_done.add('time')
            self.notes_due.append('time')
        if self.notes_due and m['note'] is not None and self._job('analyst:note', 'analyst', self.analyse, obs, 'note',
                                                                    why=self.notes_due[0]):
            self.notes_due.pop(0)
        e = self.entry
        if e and not self.trial and t - self.controller_t >= CHECK_S and self._job(
                'checker', 'checker', self.check, obs, entry=e['id']):
            self.controller_t = t
        if (e and not self.trial and self.runtime(e, t) >= VARY_AFTER_S and t >= e.get('vary_t', 0)
                and self._job('variant', 'variant', self.vary, obs, {'id': e['id'], 'desc': e['desc'], 'code': e['code']},
                              entry=e['id'])):
            e['vary_t'] = t + 30

    def _collect(self, obs, t):
        for job in [j for j in self.jobs if j['future'].done()]:
            self.jobs.remove(job)
            try:
                result = job['future'].result()
            except Exception as e:
                self.events.append((t, f"{job['role']} failed: {e!r}"))
                if job['key'] == 'writer':
                    self.writing = None
                continue
            here = job['scene'] == self.scene_id
            try:
                getattr(self, '_got_' + job['key'].replace(':', '_'))(job, result, here, obs, t)
            except Exception as e:
                self.events.append((t, f"{job['role']} result failed: {e!r}"))

    def _got_writer(self, job, code, here, obs, t):
        goal, self.writing = self.writing, None
        if not code:
            return
        try:
            entry = self._new_entry(code, t, scene=job['scene'], goal=goal)
        except Exception as e:
            self._wake(t, f'the controller written for "{goal}" failed to load: {e!r}')
            return
        if here and not self.trial:
            self._start(entry, t)
            self.events.append((t, f"writer's #{entry['id']} for \"{goal}\" started"))
        else:   # meant for another screen, or a trial runs: keep it for later
            entry['status'] = 'stored'
            self.events.append((t, f"writer's #{entry['id']} stored (written for screen #{job['scene']})"))

    def _got_variant(self, job, code, here, obs, t):
        if not code:
            return
        if here and self.entry and self.entry['id'] == job['entry'] and not self.trial:
            self._start_trial(code, t)
        else:
            entry = self._new_entry(code, t, scene=job['scene'], parent=job['entry'])
            entry['status'] = 'stored'

    def _got_checker(self, job, verdict, here, obs, t):
        if not verdict:
            return
        entry = next((e for e in self.library if e['id'] == job['entry']), None)
        trend = 'unknown'
        if entry is not None:
            if verdict.get('reading') is not None:
                entry['readings'].append((t, verdict['reading']))
            read = [v for _, v in entry['readings']]
            trend = 'unknown' if len(read) < 2 else 'rising' if read[-1] > read[-2] else 'flat or falling'
        kind = verdict.get('verdict', 'KEEP')
        self.verdicts[f'{kind} when {trend}'] += 1
        self.events.append((t, f"checker on #{job['entry']}: {kind} ({verdict.get('reason', '')[:80]})"))
        if kind == 'REPLACE' and self.entry is entry and self.writing is None:
            self.writing = f"replace #{job['entry']}: {verdict.get('reason', '')}"
            self._job('writer', 'writer', self.write, obs, self.writing)

    def _got_analyst_note(self, job, result, here, obs, t):
        if not result:
            return
        self.memory['note'], self.memory['fixes'] = result['note'], []
        self.memory['display_box'], self.box = result.get('display_box'), None
        self.events.append((t, f"analyst wrote the note ({job['why']})"))
        if job['why'] != 'start' and not self.probed:   # the game itself runs now: plan the control check
            self.probed = True
            self._job('analyst:probe', 'analyst', self.analyse, obs, 'probe')

    def _got_analyst_probe(self, job, result, here, obs, t):
        if result and here and len(result.get('probe', [])) > 1:
            self._stop(t, 'ok', 'paused for the control check')
            self.probes = [{'label': label, 'action': action and Action(action['tool'], action.get('params') or {})}
                           for label, action in result['probe']]
            self.queue, self.memory['probes'] = [], []
            self._bump()

    def _got_analyst_lesson(self, job, result, here, obs, t):
        if result and result.get('lesson'):
            self.memory['lessons'].append(result['lesson'])

    # --- the player's decisions ---

    def _apply(self, decision, t, image):
        if decision.get('keep'):
            return None
        if 'write' in decision:
            if self.writing is None:
                self.writing = decision['write']
                self._job('writer', 'writer', self.write, self._obs, self.writing)
                self.events.append((t, f'writer asked: {self.writing[:80]}'))
            return None
        if 'probe' in decision:
            self._got_analyst_probe(None, decision, True, None, t)
            return self._probe(t, image) if self.probes else None
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

    def _new_entry(self, code, t, scene=None, parent=None, goal=None):
        first = code.strip().splitlines()[0] if code.strip() else ''
        entry = {'id': len(self.library) + 1, 'scene': self.scene_id if scene is None else scene, 'code': code,
                 'parent': parent, 'goal': goal,
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
        """Per minute, how the progress number read changed while it played (0 without two readings)."""
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
        self._bump()
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
        return Action(choice['tool'], choice.get('params') or {})

    def _start_trial(self, code, t):
        parent = self.entry
        entry = self._new_entry(code, t, parent=parent['id'])
        self._start(entry, t)
        parent['status'] = 'ok'   # paused, not failed
        self.trial = {'entry': entry, 'parent': parent, 'start': t, 'end': t + TRIAL_S}
        self.events.append((t, f"trying variant #{entry['id']} of #{parent['id']}"))

    def _end_trial(self, t, image):
        trial, self.trial = self.trial, None
        variant, parent = trial['entry'], trial['parent']
        if self.entry is not variant:   # it was stopped or replaced meanwhile
            return None
        # the parent's last TRIAL_S before the trial against the variant's TRIAL_S, not a lifetime average
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
        small = np.asarray(image.convert('L').resize((NOVEL_SIZE, NOVEL_SIZE)), np.float32)
        while self.recent_thumbs and self.recent_thumbs[0][0] < t - NOVEL_WINDOW_S:
            self.recent_thumbs.popleft()
        if not self.recent_thumbs or min((np.abs(th - small) > NOVEL_PIXEL).mean()
                                         for _, th in self.recent_thumbs) >= NOVEL_SHARE:
            self.novel_t, progress = t, True
        self.recent_thumbs.append((t, small))
        box = self.memory.get('display_box')   # the progress display the analyst located, grid x0, y0, x1, y1
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
        # a changing progress display always counts as progress, repeated actions or not
        quiet = t - self.novel_t >= STUCK_S and (not box or self.ticking or t - self.display_t >= STUCK_S)
        if quiet and t - self.stuck_t >= STUCK_S and not self.probes:
            self.stuck_t = t
            if self.entry:   # a controller plays: the checker judges it now, the player is left alone
                self.entry['stuck'] = self.entry.get('stuck', 0) + 1
                self.controller_t = -float('inf')
                self.events.append((t, f"no progress signal for #{self.entry['id']}: the checker looks now"))
            else:
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
        if self.first_scene_t is None:
            self.first_scene_t = t
            self.notes_due.append('scene')    # the game itself likely shows now: revise the note
        played = old in self.play_scenes or t - self.scene_entered_t >= PLAY_S
        if played:
            self.play_scenes.add(old)
        elif self.scene_actions:          # a menu our actions led off: replay them when it shows again
            self.exits[old] = self.scene_actions[-4:]
        self.scene_actions = []
        self._stop(t, 'ok', 'the scene changed')
        self.queue = []                   # a replay that worked stops here, before it clicks into the next screen
        self.scene_id, self.scene_entered_t = new, t
        self._bump()                      # also covers the replay that may start below
        ended = t - self.attempt_t        # measured before the next attempt's clock starts
        self.attempt_t = t
        if played:                        # a play screen gave way: the attempt ended (game over or next level)
            self.attempts.append(ended)
            self.restart_t = t
            self.queue, self.queue_t = list(self.exits.get(new, [])), t + REPLAY_GAP_S
            self._job('analyst:lesson', 'analyst', self.analyse, self._obs, 'lesson')
            self._wake(t, f'game over or next level: play screen #{old} gave way to screen #{new} after {ended:.0f} s'
                          + (f'; replaying the {len(self.queue)} actions that led off this screen last time'
                             if self.queue else ''))
            return
        best = self._best(new, t)
        if best:
            self._start(best, t)
        self._wake(t, f"the scene changed (screen #{new}, {'seen before' if known else 'new'})"
                      + (f"; your best controller for it, #{best['id']}, is playing again" if best else ''))
