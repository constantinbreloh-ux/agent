"""Self-check for autopilot.py and gemma.py without the competition harness or a GPU: python test_agent.py"""
import io
import sys
import time
import types
from dataclasses import dataclass, field

from PIL import Image

protocol = types.ModuleType('gga.protocol')   # stand-in for the harness's gga.protocol


@dataclass
class Action:
    tool: str
    params: dict = field(default_factory=dict)


protocol.Action = Action
sys.modules['gga'] = types.ModuleType('gga')
sys.modules['gga.protocol'] = protocol

import autopilot  # noqa: E402
from autopilot import Agent  # noqa: E402

CONTROLLER = '''
def control(frame, mem):
    mem['n'] = mem.get('n', 0) + 1
    if mem['n'] > 5:
        mem['handoff'] = 'done dodging'
        return None
    assert frame.shape == (1000, 1000, 3)
    x = frame[:, :, 0].mean(axis=0).argmax()   # brightest red column, a numpy int as models write it
    return {"tool": "click", "params": {"x": x, "y": 500.4}}   # grid coordinates
'''
MOVE = '# skill: hold the pointer mid-screen\ndef control(frame, mem): return {"tool": "move", "params": {"x": 500, "y": 500}}'


class FakeModel(Agent):
    """Every role answers from the test: player replies (a list or a function), the others from attributes."""
    def __init__(self, replies):
        self.replies, self.seen = (replies if callable(replies) else list(replies)), []
        self.write_code = self.variant_code = self.verdict = None
        self.note = self.probe_plan = None
        self.write_s, self.think_s = 0.3, 0.3
        super().__init__()

    def think(self, obs, event, view):
        self.seen.append(event)
        time.sleep(self.think_s)   # a slow model
        if callable(self.replies):
            return self.replies(self, event)
        return self.replies.pop(0) if self.replies else {'tool': 'skip', 'params': {}}

    def write(self, obs, goal, view):
        time.sleep(self.write_s)
        return self.write_code

    def vary(self, obs, entry, view):
        return self.variant_code

    def check(self, obs, view):
        return self.verdict

    def analyse(self, obs, task, view):
        if task == 'note' and self.note:
            return {'note': self.note, 'display_box': None}
        if task == 'probe' and self.probe_plan:
            return {'probe': self.probe_plan}
        return None


def obs(t, red_x=30, color=(0, 0, 0), program_left_s=0.0, last_action=None):
    img = Image.new('RGB', (64, 48), color)
    img.paste((255, 0, 0), (red_x, 0, red_x + 1, 48))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=95)
    return types.SimpleNamespace(elapsed_s=t, image=buf.getvalue(), program_left_s=program_left_s,
                                 last_action=last_action or {})


def play(agent, seconds, start=0.0, **kw):
    actions, t = [], start
    while t < start + seconds:
        began = time.time()
        actions.append(agent.step(obs(t, **kw), []))
        assert time.time() - began < 0.05, 'step() blocked'
        time.sleep(0.2)
        t += 0.2
    return actions


def texts(agent):
    return [e for _, e in agent.events]


# 1. The controller plays without the model, hands back, then a crash wakes the player.
agent = FakeModel([{'controller': CONTROLLER}, {'controller': 'def control(frame, mem): 1/0'}])
actions = play(agent, 2.4)
clicks = [a for a in actions if a.tool == 'click']
assert len(clicks) == 5 and all(type(a.params['x']) is int and a.params['y'] == 23 and abs(a.params['x'] - 30) <= 1 for a in clicks), actions
assert agent.seen[:3] == ['new game', 'done dodging', 'controller crashed: ZeroDivisionError(\'division by zero\')'], agent.seen

# 2. A sequence keeps the player asleep until it has played out.
agent = FakeModel([{'tool': 'sequence', 'params': {'steps': [{'keys': ['a'], 'duration_ms': 1500}]}}])
play(agent, 0.4)
play(agent, 1.2, start=0.4, program_left_s=1.0)   # the game reports the sequence still playing
assert agent.seen == ['new game'], agent.seen

# 3. A new game (clock back near 0) resets per-game state and wakes the player.
agent.memory['note'] = 'old game'
play(agent, 0.4, start=0.0)
assert agent.memory['note'] is None and agent.library == [] and agent.seen[-1] == 'new game', (agent.memory, agent.seen)

# 4. A lasting scene change stops the controller; a controller played there, so it ended an attempt.
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "skip"}'}])
play(agent, 1.4)
play(agent, 1.6, start=1.4, color=(200, 200, 200))
assert agent.controller is None and any(e.startswith('game over or next level: play screen #0 gave way to screen #1') for e in agent.seen), agent.seen

# 5. A refused controller action stops the controller and tells the player why.
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "click", "params": {"x": 1, "y": 1}}'}])
play(agent, 0.6)
play(agent, 0.4, start=0.6, last_action={'tool': 'click', 'ok': False, 'error': 'x out of range'})
assert agent.controller is None and "the game refused the controller's action: x out of range" in agent.seen, agent.seen
print('core ok')

# 6. Stage 1, the writer: WRITE goes to the writer in the background, the player keeps playing meanwhile,
#    and the finished controller starts by itself because its screen still shows.
def player(agent, event):
    if not agent.library and agent.writing is None and not getattr(agent, 'asked', False):
        agent.asked = True
        return {'write': 'hold the pointer mid-screen'}
    return {'tool': 'press_key', 'params': {'key': 'a'}}
agent = FakeModel(player)
agent.write_code, agent.write_s, agent.think_s = MOVE, 1.2, 0.8   # a reply is in flight when the controller starts
actions = play(agent, 3.4)
assert 'writer asked: hold the pointer mid-screen' in texts(agent), texts(agent)
assert any(e.startswith("writer's #1") and e.endswith('started') for e in texts(agent)), texts(agent)
assert agent.entry is agent.library[0] and agent.library[0]['goal'] == 'hold the pointer mid-screen', agent.library
assert agent.stats['steps while writing'] > 0 and agent.stats['skips while writing'] < agent.stats['steps while writing'], agent.stats
assert any(a.tool == 'press_key' for a in actions) and any(a.tool == 'move' for a in actions), actions   # played on
# the player's reply in flight when the controller started is from an older decision count: dropped, not played
assert agent.stats['player replies dropped'] >= 1 and agent.entry['status'] == 'running', (agent.stats, texts(agent))
assert actions[-1].tool == 'move', actions[-3:]   # the controller still plays at the end
print('writer ok')

# 6b. A controller finished after its screen went away is stored, not started.
agent = FakeModel(lambda agent, event: {'write': 'something'} if not agent.library and agent.writing is None
                  else {'tool': 'wait', 'params': {'duration_ms': 100}})
agent.write_code, agent.write_s = MOVE, 2.0
play(agent, 1.2)
play(agent, 2.6, start=1.2, color=(200, 200, 200))
assert agent.library and agent.library[0]['status'] == 'stored' and agent.controller is None, (agent.library, texts(agent))
print('stored ok')

# 7. Stage 2, the checker: every CHECK_S while a controller plays; REPLACE sends the writer a goal, and its
#    controller takes over. The verdict and the number read are counted.
autopilot.CHECK_S = 1
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "hold_keys", "params": {"keys": ["", "a"], "duration_ms": 200}}'}])
agent.verdict = {'verdict': 'REPLACE', 'reason': 'it never aims', 'reading': 5.0}
agent.write_code, agent.write_s = MOVE, 0.3
actions = play(agent, 3.4)
held = [a for a in actions if a.tool == 'hold_keys']
assert held and all(a.params['keys'] == ['a'] for a in held), actions   # empty key dropped
assert any(e.startswith('checker on #1: REPLACE') for e in texts(agent)), texts(agent)
assert agent.library[0]['readings'] and agent.verdicts['REPLACE when unknown'] >= 1, (agent.library[0], agent.verdicts)
assert agent.entry is agent.library[1] and agent.library[1]['goal'] == 'replace #1: it never aims', agent.library
autopilot.CHECK_S = 10
print('checker ok')

# 8. Stage 2, the analyst: the note at the start, a revision at the first scene change, then the control check;
#    the player is not held up meanwhile. The time from the first scene change to the first action is measured.
agent = FakeModel(lambda agent, event: {'tool': 'wait', 'params': {'duration_ms': 100}})
agent.note = 'Type: test\nProgress display: none'
agent.probe_plan = [('no input', None), ('Q', {'tool': 'press_key', 'params': {'key': 'q'}})]
play(agent, 1.4)
assert agent.memory['note'] == agent.note and 'analyst wrote the note (start)' in texts(agent), texts(agent)
actions = play(agent, 3.0, start=1.4, color=(200, 200, 200))
assert 'analyst wrote the note (scene)' in texts(agent), texts(agent)
assert any(a.tool == 'press_key' for a in actions), actions
assert agent.memory['probes'] == ['no input: nothing visible changed', 'Q: nothing visible changed'], agent.memory['probes']
assert agent.first_play_delay is not None and agent.first_play_delay < 2, agent.first_play_delay
print('analyst ok')

# 9. Menu -> (action) -> play -> (action) -> game over -> (action) -> play: one game over, an attempt longer
#    than 0 s, and on the way back into play the best controller restarts without the player.
autopilot.BEST_MIN_S = 0.5
MENU, PLAY, OVER = (0, 0, 0), (200, 200, 200), (0, 0, 220)
def policy(agent, event):
    if agent.scene_id == 1:   # play: one action of its own, then a controller; later KEEP
        if not getattr(agent, 'acted', False):
            agent.acted = True
            return {'tool': 'press_key', 'params': {'key': 'a'}}
        return {'keep': True} if agent.library else {'controller': MOVE}
    return {'tool': 'click', 'params': {'x': 3 + agent.scene_id, 'y': 3}}   # menu and game over: a button
agent = FakeModel(policy)
play(agent, 1.6, color=MENU)
play(agent, 3.4, start=1.6, color=PLAY)
assert agent.entry and agent.entry['desc'] == 'hold the pointer mid-screen' and 1 in agent.play_scenes, agent.events
agent.library[0]['events'] = 3   # stand-in for progress signals while it played
play(agent, 2.0, start=5.0, color=OVER)
play(agent, 2.0, start=7.0, color=PLAY)
overs = [e for e in texts(agent) if e.startswith('game over')]
assert len(overs) == 1 and 'play screen #1 gave way to screen #2' in overs[0], agent.events
assert len(agent.attempts) == 1 and agent.attempts[0] > 2, agent.attempts
assert 1 not in agent.exits and 0 in agent.exits and 2 in agent.exits, agent.exits   # only menus have exits
assert any('your best controller for it, #1, is playing again' in e for e in texts(agent)), agent.events
assert agent.entry is agent.library[0] and agent.restart_delays, (agent.entry, agent.restart_delays)
print('game over ok')

# 9b. A second death replays the button that led off the game-over screen, and the replay stops once play is back.
agent.library[0]['events'] = 3
play(agent, 1.6, start=9.0, color=OVER)
play(agent, 1.6, start=10.6, color=PLAY)
assert agent.queue == [] and agent.scene_id == 1, agent.queue
overs = [e for e in texts(agent) if e.startswith('game over')]
assert len(overs) == 2 and 'replaying the 4 actions that led off this screen' in overs[1], overs   # up to the last 4
print('replay ok')

# 9c. A display that changes every second is a clock: not progress.
agent = FakeModel(lambda agent, event: {'tool': 'wait', 'params': {'duration_ms': 100}})
agent.step(obs(0.0), [])   # the first step starts the game and resets memory
agent.memory['display_box'] = (0, 0, 999, 999)
for i in range(1, 14):
    agent.step(obs(float(i), red_x=5 + 4 * i % 50, color=(0, 0, 0) if i % 2 else (90, 90, 90)), [])
assert agent.ticking is True and 'like a clock' in agent.memory['signals'], agent.memory['signals']   # a plain bool: json needs it
print('ticking ok')

# 10. Stage 0: new screen content counts against the last NOVEL_WINDOW_S only, so a screen comes back as new.
autopilot.NOVEL_WINDOW_S = 2
agent = Agent()
for i, color in enumerate([(0, 0, 0)] * 2 + [(200, 200, 200)] * 3 + [(0, 0, 0)]):
    agent.step(obs(float(i), color=color), [])
assert agent.novel_t == 5.0, agent.novel_t   # black again after 3 s away: new within the window
autopilot.NOVEL_WINDOW_S = 30

# 10a. Calibrated on the run videos: a thin moving line is new content, the same screen is not.
agent = Agent()
for i in range(4):
    agent.step(obs(float(i), red_x=30), [])
assert agent.novel_t == 0.0, agent.novel_t
agent.step(obs(4.0, red_x=10), [])
assert agent.novel_t == 4.0, agent.novel_t

# 10b. No progress: a static screen and repeated actions wake the player with the measurements.
autopilot.STUCK_S = 2
agent = FakeModel(lambda agent, event: {'tool': 'click', 'params': {'x': 3, 'y': 3}})
play(agent, 5.0)
stuck = [e for e in texts(agent) if e.startswith('no progress signal')]
assert stuck and 'of your last 10 actions were click' in stuck[-1] and 'no progress display located' in stuck[-1], agent.events
assert agent.stats['idle steps'] > 0 and agent.stats['idle steps'] < agent.stats['steps'], agent.stats

# 10c. While a controller plays, no progress goes to the checker, not the player.
agent = FakeModel([{'controller': MOVE}])
agent.verdict = {'verdict': 'KEEP', 'reason': 'fine', 'reading': None}
play(agent, 5.0)
assert 'no progress signal for #1: the checker looks now' in texts(agent), texts(agent)
assert not any(e.startswith('no progress signal:') for e in agent.seen), agent.seen
assert any(e.startswith('checker on #1: KEEP') for e in texts(agent)) and agent.entry is agent.library[0], texts(agent)
autopilot.STUCK_S = 15
print('signals ok')

# 11. A variant that makes no more progress than its parent is dropped, and the parent plays again.
autopilot.VARY_AFTER_S, autopilot.TRIAL_S = 0.6, 0.6
agent = FakeModel([{'controller': MOVE}])
agent.variant_code = '# skill: the same, a bit left\ndef control(frame, mem): return {"tool": "move", "params": {"x": 400, "y": 500}}'
play(agent, 2.6)
assert 'trying variant #2 of #1' in texts(agent) and 'variant #2 is not better than #1: back to #1' in texts(agent), texts(agent)
assert agent.entry is agent.library[0] and agent.library[1]['status'] == 'worse than #1', agent.library
assert agent.variants == {'dropped': 1}, agent.variants
autopilot.VARY_AFTER_S, autopilot.TRIAL_S = 20, 15
print('variant ok')

# 12. step() never raises.
agent = Agent()
bad = obs(0.0)
bad.image = b'not a jpeg'
assert agent.step(bad, []).tool == 'skip' and agent.events[-1][1].startswith('agent error'), agent.events
print('no raise ok')

# 13. Gemma: parsing.
import http.server, json, threading  # noqa: E401,E402
import gemma  # noqa: E402

assert gemma.parse('{"tool":"click","params":{"x":999,"y":0}}', 641, 481) == {'tool': 'click', 'params': {'x': 640, 'y': 0}}
seq = gemma.parse('Sure: {"tool":"sequence","params":{"steps":[{"keys":[],"buttons":[],"x":500,"y":500,"duration_ms":0}]}}', 2000, 2000)
assert seq['params']['steps'][0]['x'] == 1000, seq
assert gemma.parse('```python\ndef control(frame, mem):\n    return None\n```', 1, 1) == {'controller': 'def control(frame, mem):\n    return None\n'}
assert gemma.parse('Progress: alive, 3 lives\nKEEP', 1, 1) == {'keep': True}
assert gemma.parse('Progress: alive\nREUSE #3', 1, 1) == {'reuse': 3}
assert gemma.parse('Progress: 40\nWRITE: dodge the red shots, fire at green ships', 1, 1) == {'write': 'dodge the red shots, fire at green ships'}
assert gemma.parse('{"tool":"drag","params":{"x":0,"y":0,"x2":999,"y2":999}}', 641, 481)['params'] == {'x': 0, 'y': 0, 'x2': 640, 'y2': 480}
try:
    gemma.parse('I will press space', 1, 1)
    raise AssertionError('should fail')
except ValueError:
    pass
probes = gemma.parse_probe('Q (build): {"tool":"press_key","params":{"key":"q"}}\nleft click: {"tool":"click","params":{"x":999,"y":0}}\nnonsense', 641, 481)
assert [p[0] for p in probes] == ['no input (the screen alone)', 'Q (build)', 'left click'] and probes[2][1]['params']['x'] == 640, probes
assert gemma.display_box('Progress display: 0,930,999,999') == (0, 930, 999, 999)
assert gemma.display_box('Progress display: none\nCorrection: Progress display: (10, 900, 300, 990)') == (10, 900, 300, 990)
assert gemma.display_box('Progress display: none') is None
assert gemma.first_number('score 1,250 dollars') == 1250.0 and gemma.first_number('alive') is None
print('parse ok')

# 14. Gemma's requests to the vLLM server, against a fake one: roles, priorities, images, the log.
class FakeServer(http.server.BaseHTTPRequestHandler):
    reply = '{"tool":"click","params":{"x":0,"y":999}}'
    def do_POST(self):
        FakeServer.body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        out = json.dumps({'choices': [{'message': {'content': FakeServer.reply}}], 'usage': {'completion_tokens': 20}}).encode()
        self.send_response(200), self.end_headers(), self.wfile.write(out)
    def log_message(self, *args): pass

server = http.server.HTTPServer(('127.0.0.1', 0), FakeServer)
threading.Thread(target=server.serve_forever, daemon=True).start()
gemma.SERVER = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
agent = gemma.Agent()
o = obs(5.0)
o.info = {'description': 'd', 'controls': 'c', 'width': 64, 'height': 48, 'remaining_s': 9}
FakeServer.reply = 'Progress: score 1,250\nNote: Progress display: 0,900,999,999\n{"tool":"click","params":{"x":0,"y":999}}'
decision = agent.think(o, 'ready', agent._view(5.0))
assert decision['tool'] == 'click' and decision['params'] == {'x': 0, 'y': 47}, decision
assert decision['updates']['progress'] == ('score 1,250', 1250.0) and decision['updates']['display_box'] == (0, 900, 999, 999), decision
assert FakeServer.body['priority'] == 0, FakeServer.body   # the player goes first
user = FakeServer.body['messages'][1]['content']
assert user[0]['image_url']['url'].startswith('data:image/jpeg;base64,') and 'ready' in user[1]['text'], user
assert agent.replies[-1][5] == 'player' and agent.replies[-1][6] == 1, agent.replies[-1]
agent.entry = {'id': 1, 'readings': []}
agent._update(decision['updates'], 5.0)
assert agent.entry['readings'] == [(5.0, 1250.0)] and agent.memory['display_box'] == (0, 900, 999, 999), agent.memory
agent.entry = None

agent.last_frames.extend([(i, o.image) for i in range(8)])
FakeServer.reply = 'Lesson: a red shot hit the ship; dodge earlier'
assert agent.analyse(o, 'lesson', agent._view(5.0)) == {'lesson': 'a red shot hit the ship; dodge earlier'}
content = FakeServer.body['messages'][1]['content']
assert [c['type'] for c in content] == ['image_url'] * 6 + ['text'] and FakeServer.body['priority'] == 1, content
FakeServer.reply = 'Type: shooter\nProgress display: 380,0,420,40\nWhat counts: money'
assert agent.analyse(o, 'note', agent._view(5.0)) == {'note': FakeServer.reply, 'display_box': (380, 0, 420, 40)}
FakeServer.reply = '```python\n# skill: aim\ndef control(frame, mem): return None\n```'
assert agent.write(o, 'aim at ships', agent._view(5.0)).startswith('# skill: aim') and FakeServer.body['max_tokens'] == 1500
agent.controller, agent.controller_start = True, 1.0
agent.entry = {'id': 3, 'desc': 'aim', 'code': 'def control(frame, mem): pass\n', 'readings': [(2.0, 10.0)]}
FakeServer.reply = 'Progress: 40\nVerdict: REPLACE\nReason: it shoots at asteroids, not ships'
assert agent.check(o, agent._view(5.0)) == {'verdict': 'REPLACE', 'reason': 'it shoots at asteroids, not ships', 'reading': 40.0}
print('gemma ok')
