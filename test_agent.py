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
    x = frame[:, :, 0].mean(axis=0).argmax()   # brightest red column, a numpy int as models write it
    return {"tool": "click", "params": {"x": x, "y": 10.4}}
'''


class FakeModel(Agent):
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []
        super().__init__()

    def think(self, obs, event, memory):
        self.seen.append(event)
        time.sleep(0.3)   # a slow model
        return self.replies.pop(0) if self.replies else {'tool': 'skip', 'params': {}}


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
assert len(clicks) == 5 and all(type(a.params['x']) is int and a.params['y'] == 10 and abs(a.params['x'] - 30) <= 1 for a in clicks), actions
assert agent.seen[:3] == ['new game', 'done dodging', 'controller crashed: ZeroDivisionError(\'division by zero\')'], agent.seen

# 2. A sequence keeps the model asleep until it has played out.
agent = FakeModel([{'tool': 'sequence', 'params': {'steps': [{'keys': ['a'], 'duration_ms': 1500}]}}])
play(agent, 0.4)
play(agent, 1.2, start=0.4, program_left_s=1.0)   # the game reports the sequence still playing
assert agent.seen == ['new game'], agent.seen

# 3. A new game (clock back near 0) resets per-game state and wakes the model.
agent.memory['note'] = 'old game'
play(agent, 0.4, start=0.0)
assert agent.memory == {} and agent.seen[-1] == 'new game', (agent.memory, agent.seen)

# 4. A lasting scene change stops the controller and wakes the model.
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "skip"}'}])
play(agent, 1.4)
play(agent, 1.6, start=1.4, color=(200, 200, 200))
assert agent.controller is None and 'the scene changed' in agent.seen, agent.seen

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
assert agent.think(o, 'ready', {}) == {'tool': 'click', 'params': {'x': 0, 'y': 47}}
user = FakeServer.body['messages'][1]['content']
assert user[0]['image_url']['url'].startswith('data:image/jpeg;base64,') and 'ready' in user[1]['text'], user
assert agent.replies[-1][4] == 20
print('vllm request ok')

# 8. Check-ins: a running controller wakes the model every CHECK_S, plays on meanwhile, and KEEP keeps it.
import autopilot  # noqa: E402
autopilot.CHECK_S = 1
agent = FakeModel([{'controller': 'def control(frame, mem): return {"tool": "hold_keys", "params": {"keys": ["", "a"], "duration_ms": 200}}'},
                   {'keep': True}])
actions = play(agent, 2.6)
assert any(e.startswith('check-in: your controller has played for') for e in agent.seen), agent.seen
assert agent.controller is not None and agent.controller_code.startswith('def control'), agent.seen
held = [a for a in actions if a.tool == 'hold_keys']
assert len(held) >= 10 and all(a.params['keys'] == ['a'] for a in held), actions   # played on, empty key dropped
autopilot.CHECK_S = 10
assert gemma.parse('Score: 40\nKEEP', 1, 1) == {'keep': True}
print('check-in ok')
