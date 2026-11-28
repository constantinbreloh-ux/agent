"""Self-check for agent.py without the competition harness: python test_agent.py"""
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


class FakeModel(Agent):
    def __init__(self, replies):
        self.replies, self.seen = (replies if callable(replies) else list(replies)), []
        super().__init__()

    def think(self, obs, event, memory):
        self.seen.append(event)
        time.sleep(0.3)   # a slow model
        if callable(self.replies):
            return self.replies(self, event)
        return self.replies.pop(0) if self.replies else {'tool': 'skip', 'params': {}}

    def vary(self, obs, entry, memory):
        return getattr(self, 'variant_code', None)


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


# 1. The controller plays without the model, hands back, then a crash wakes the model.
agent = FakeModel([{'controller': CONTROLLER}, {'controller': 'def control(frame, mem): 1/0'}])
actions = play(agent, 2.4)
clicks = [a for a in actions if a.tool == 'click']
assert len(clicks) == 5 and all(type(a.params['x']) is int and a.params['y'] == 23 and abs(a.params['x'] - 30) <= 1 for a in clicks), actions
assert agent.seen[:3] == ['new game', 'done dodging', 'controller crashed: ZeroDivisionError(\'division by zero\')'], agent.seen

# 2. A sequence keeps the model asleep until it has played out.
agent = FakeModel([{'tool': 'sequence', 'params': {'steps': [{'keys': ['a'], 'duration_ms': 1500}]}}])
play(agent, 0.4)
play(agent, 1.2, start=0.4, program_left_s=1.0)   # the game reports the sequence still playing
assert agent.seen == ['new game'], agent.seen

# 3. A new game (clock back near 0) resets per-game state and wakes the model.
agent.memory['note'] = 'old game'
play(agent, 0.4, start=0.0)
assert 'note' not in agent.memory and agent.library == [] and agent.seen[-1] == 'new game', (agent.memory, agent.seen)

# 4. A lasting scene change stops the controller and wakes the model; a controller played there, so it ended an attempt.
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "skip"}'}])
play(agent, 1.4)
play(agent, 1.6, start=1.4, color=(200, 200, 200))
assert agent.controller is None and any(e.startswith('game over or next level: play screen #0 gave way to screen #1') for e in agent.seen), agent.seen

print('ok')

# 5. A refused controller action stops the controller and tells the model why.
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "click", "params": {"x": 1, "y": 1}}'}])
play(agent, 0.6)
play(agent, 0.4, start=0.6, last_action={'tool': 'click', 'ok': False, 'error': 'x out of range'})
assert agent.controller is None and "the game refused the controller's action: x out of range" in agent.seen, agent.seen

# 6. Parsing Gemma's replies.
from gemma import parse  # noqa: E402

assert parse('{"tool":"click","params":{"x":999,"y":0}}', 641, 481) == {'tool': 'click', 'params': {'x': 640, 'y': 0}}
seq = parse('Sure: {"tool":"sequence","params":{"steps":[{"keys":[],"buttons":[],"x":500,"y":500,"duration_ms":0}]}}', 2000, 2000)
assert seq['params']['steps'][0]['x'] == 1000, seq
assert parse('```python\ndef control(frame, mem):\n    return None\n```', 1, 1) == {'controller': 'def control(frame, mem):\n    return None\n'}
assert 'control' in parse('```python\ndef control(frame, mem):\n    return No', 1, 1)['controller']   # cut off by max tokens
try:
    parse('I will press space', 1, 1)
    raise AssertionError('should fail')
except ValueError:
    pass

print('parse ok')

# 7. Gemma's request to the vLLM server, against a fake server.
import http.server, json, threading  # noqa: E401,E402
import gemma  # noqa: E402

class FakeServer(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        FakeServer.body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        out = json.dumps({'choices': [{'message': {'content': '{"tool":"click","params":{"x":0,"y":999}}'}}],
                          'usage': {'completion_tokens': 20}}).encode()
        self.send_response(200), self.end_headers(), self.wfile.write(out)
    def log_message(self, *args): pass

server = http.server.HTTPServer(('127.0.0.1', 0), FakeServer)
threading.Thread(target=server.serve_forever, daemon=True).start()
gemma.SERVER = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
agent = gemma.Agent()
o = obs(1.0)
o.info = {'description': 'd', 'controls': 'c', 'width': 64, 'height': 48, 'remaining_s': 9}
assert agent.think(o, 'ready', agent.memory) == {'tool': 'click', 'params': {'x': 0, 'y': 47}}
user = FakeServer.body['messages'][1]['content']
assert user[0]['image_url']['url'].startswith('data:image/jpeg;base64,') and 'ready' in user[1]['text'], user
assert agent.replies[-1][4] == 20 and agent.replies[0][1] == 'game note', agent.replies
print('vllm request ok')

# 8. Check-ins: a running controller wakes the model every CHECK_S, plays on meanwhile, and KEEP keeps it.
import autopilot  # noqa: E402
autopilot.CHECK_S = 1
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "hold_keys", "params": {"keys": ["", "a"], "duration_ms": 200}}'},
                   {'keep': True}])
actions = play(agent, 2.6)
assert any(e.startswith('check-in: your controller has played for') for e in agent.seen), agent.seen
assert agent.controller is not None and agent.entry['code'].startswith('def control'), agent.seen
held = [a for a in actions if a.tool == 'hold_keys']
assert len(held) >= 10 and all(a.params['keys'] == ['a'] for a in held), actions   # played on, empty key dropped
autopilot.CHECK_S = 10
assert gemma.parse('Progress: alive, 3 lives\nKEEP', 1, 1) == {'keep': True}
print('check-in ok')
assert gemma.parse('{"tool":"drag","params":{"x":0,"y":0,"x2":999,"y2":999}}', 641, 481)['params'] == {'x': 0, 'y': 0, 'x2': 640, 'y2': 480}
print('drag ok')

# 9. Menu -> (action) -> play -> (action) -> game over -> (action) -> play: one game over, an attempt longer
#    than 0 s, and on the way back into play the best controller restarts without the model.
autopilot.BEST_MIN_S = 0.5
MENU, PLAY, OVER = (0, 0, 0), (200, 200, 200), (0, 0, 220)
MOVE = '# skill: hold the pointer mid-screen\ndef control(frame, mem): return {"tool": "move", "params": {"x": 500, "y": 500}}'
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
overs = [e for _, e in agent.events if e.startswith('game over')]
assert len(overs) == 1 and 'play screen #1 gave way to screen #2' in overs[0], agent.events
assert len(agent.attempts) == 1 and agent.attempts[0] > 2, agent.attempts
assert 1 not in agent.exits and 0 in agent.exits and 2 in agent.exits, agent.exits   # only menus have exits
assert any('your best controller for it, #1, is playing again' in e for _, e in agent.events), agent.events
assert agent.entry is agent.library[0] and agent.restart_delays, (agent.entry, agent.restart_delays)
print('game over ok')

# 9b. A second death replays the button that led off the game-over screen.
agent.library[0]['events'] = 3
play(agent, 1.6, start=9.0, color=OVER)
play(agent, 1.6, start=10.6, color=PLAY)
assert agent.queue == [] and agent.scene_id == 1, agent.queue   # the replay stopped once play was back
overs = [e for _, e in agent.events if e.startswith('game over')]
assert len(overs) == 2 and 'replaying the 4 actions that led off this screen' in overs[1], overs   # up to the last 4
print('replay ok')

# 9c. A display that changes every second is a clock: not progress.
agent = FakeModel(lambda agent, event: {'tool': 'wait', 'params': {'duration_ms': 100}})
agent.step(obs(0.0), [])   # the first step starts the game and resets memory
agent.memory['display_box'] = (0, 0, 999, 999)
for i in range(1, 14):
    agent.step(obs(float(i), red_x=5 + 4 * i % 50, color=(0, 0, 0) if i % 2 else (90, 90, 90)), [])
assert agent.ticking and 'like a clock' in agent.memory['signals'], agent.memory['signals']
print('ticking ok')

# 10. No progress: a static screen and repeated actions wake the model with the measurements.
autopilot.STUCK_S = 2
agent = FakeModel(lambda agent, event: {'tool': 'click', 'params': {'x': 3, 'y': 3}})
play(agent, 5.0)
stuck = [e for _, e in agent.events if e.startswith('no progress signal')]
assert stuck and 'of your last 10 actions were click' in stuck[-1] and 'no progress display located' in stuck[-1], agent.events
autopilot.STUCK_S = 15
print('stuck ok')

# 11. Control check: each action once, the screen watched, the results for the model.
agent = FakeModel([{'probe': [('no input', None), ('Q', {'tool': 'press_key', 'params': {'key': 'q'}})]}])
actions = play(agent, 2.6)
assert any(a.tool == 'press_key' for a in actions), actions
assert agent.memory['probes'] == ['no input: nothing visible changed', 'Q: nothing visible changed'], agent.memory
assert 'control check done: see its results' in agent.seen, agent.seen
print('probe ok')

# 12. A variant that makes no more progress than its parent is dropped, and the parent plays again.
autopilot.VARY_AFTER_S, autopilot.TRIAL_S = 0.6, 0.6
agent = FakeModel([{'controller': MOVE}])
agent.variant_code = '# skill: the same, a bit left\ndef control(frame, mem): return {"tool": "move", "params": {"x": 400, "y": 500}}'
play(agent, 2.6)
texts = [e for _, e in agent.events]
assert 'trying variant #2 of #1' in texts and 'variant #2 is not better than #1: back to #1' in texts, texts
assert agent.entry is agent.library[0] and agent.library[1]['status'] == 'worse than #1', agent.library
assert agent.variants == {'dropped': 1}, agent.variants
autopilot.VARY_AFTER_S, autopilot.TRIAL_S = 20, 15
print('variant ok')

# 13. step() never raises.
agent = Agent()
bad = obs(0.0)
bad.image = b'not a jpeg'
assert agent.step(bad, []).tool == 'skip' and agent.events[-1][1].startswith('agent error'), agent.events
print('no raise ok')

# 14. Gemma side: REUSE, control-check lines, the progress display box, game-over images.
assert gemma.parse('Progress: alive\nREUSE #3', 1, 1) == {'reuse': 3}
probes = gemma.parse_probe('Q (build): {"tool":"press_key","params":{"key":"q"}}\nleft click: {"tool":"click","params":{"x":999,"y":0}}\nnonsense', 641, 481)
assert [p[0] for p in probes] == ['no input (the screen alone)', 'Q (build)', 'left click'] and probes[2][1]['params']['x'] == 640, probes
assert gemma.display_box('Progress display: 0,930,999,999') == (0, 930, 999, 999)
assert gemma.display_box('Progress display: none\nCorrection: Progress display: (10, 900, 300, 990)') == (10, 900, 300, 990)
assert gemma.display_box('Progress display: none') is None
agent = gemma.Agent()
o = obs(5.0)
o.info = {'description': 'd', 'controls': 'c', 'width': 64, 'height': 48, 'remaining_s': 9}
agent.memory['last_frames'].extend([(i, o.image) for i in range(8)])
agent.think(o, 'game over: back on screen #0 after 40 s', agent.memory)
content = FakeServer.body['messages'][1]['content']
assert [c['type'] for c in content] == ['image_url'] * 6 + ['text'] and 'last seconds before the game over' in content[-1]['text'], content
o.elapsed_s = 25.0
agent.think(o, 'ready', agent.memory)   # 20 s: the note is revised, a control check planned
assert agent.memory['note_at_20s'] and agent.memory['probed'], agent.memory.keys()
agent.controller, agent.controller_start = True, 20.0
agent.entry = {'id': 1, 'code': 'def control(frame, mem): pass\n', 'readings': []}
gemma.chat = lambda messages, max_tokens: ('Progress: score 1,250\nKEEP', 5)
assert agent.think(o, 'check-in', agent.memory) == {'keep': True} and agent.entry['readings'] == [(25.0, 1250.0)]
print('gemma ok')
