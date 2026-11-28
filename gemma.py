# Notebook cell: Gemma, served by vLLM on localhost (setup cell), as the model behind autopilot.Agent.
import base64
import io
import json
import re
import time
import urllib.request

from PIL import Image

from autopilot import Agent as AutoPilot, to_pixels

SERVER = 'http://localhost:8000/v1/chat/completions'

TOOLS = '''Tools. Coordinates are on a 0-999 grid over the screen (x: left to right, y: top to bottom).
Durations are in ms, at most 10000. Keys: a printable character or a name such as ArrowLeft, Space, Enter,
Shift, Escape. Every reply except skip replaces what is still playing.
{"tool":"click","params":{"x":500,"y":500}}                     click; optional "button": "left", "middle" or "right"
{"tool":"move","params":{"x":500,"y":500}}                      move the pointer without clicking (aiming, hovering)
{"tool":"drag","params":{"x":100,"y":500,"x2":800,"y2":500,"hold_ms":100}}   press, move to x2,y2, release
{"tool":"scroll","params":{"x":500,"y":500,"dy":100}}           mouse wheel; dy > 0 down, 100 = one notch; optional dx
{"tool":"press_key","params":{"key":"Space","hold_ms":100}}     tap a key; hold it longer to keep moving
{"tool":"hold_keys","params":{"keys":["ArrowRight","Space"],"duration_ms":500}}   hold keys together, then release
{"tool":"key_down","params":{"key":"ArrowRight"}}               hold a key until key_up
{"tool":"key_up","params":{"key":"ArrowRight"}}                 release a key held by key_down
{"tool":"wait","params":{"duration_ms":500}}                    stop what is playing and idle
{"tool":"skip","params":{}}                                     change nothing: what is playing plays on
{"tool":"sequence","params":{"steps":[{"keys":["ArrowRight"],"buttons":[],"x":500,"y":500,"duration_ms":400},{"keys":["ArrowUp","x"],"buttons":["left"],"x":500,"y":500,"duration_ms":200}]}}
    a timed program of up to 200 steps that plays while you think. Each step holds exactly its keys and
    buttons (all else is released) for duration_ms; x, y are optional. The last step stays held.'''

RULES = f'''You play a real-time browser game from screenshots. The game keeps running while you think, and each
of your replies takes seconds, so prefer moves that keep playing on their own.

{TOOLS}

Start every reply with one line: Progress: <what counts in this game (your game note), as it stands on this
screen, e.g. "score 120", "alive, 3 lives", "level 2", "distance 450", "timer 1:20", or "unknown">
Then reply in exactly one of these three forms.

1. One action, as JSON on one line, from the tools above.

2. A controller: Python code that plays every 200 ms from the screen without you. Use it when the game
   needs fast reactions (dodging, aiming, steering, timing jumps).
```python
def control(frame, mem):
    # frame: numpy uint8 array (1000, 1000, 3), RGB: the screen scaled to the 0-999 grid, so frame[y, x]
    #        is grid point (x, y). np.argwhere returns (y, x). np is numpy.
    # mem: a dict kept between calls.
    # Return one action dict from the tools above, in grid coordinates as they come from frame,
    # or set mem["handoff"] = "<why>" and return None to give control back to you.
    ...
```
   Rules for controllers:
   - Use the coordinates from frame as they are: frame already is the grid, never rescale them.
   - Act on what you detect: aim at the targets you found, move away from the threats you found.
     Code that finds things and then ignores them is useless.
   - Serve what counts in this game (your game note): survive, score, go fast or get far.
   - No comments, no docstrings, no blank lines: every token you write costs game time.
   - Fast, well under 0.1 s per call: numpy color masks, never Python loops over pixels.
   - Give control back when something happens it cannot handle.

3. KEEP
   Only while your controller is playing and the progress shows it works: it plays on unchanged.
   If it is not making progress on what counts, reply with a better controller instead.

Prefer a sequence when it is enough. Test an uncertain control briefly with one action before you rely on it.
If a menu or game-over screen shows, start the game the way the brief or the screen says.
If something you tried did not help, try something different.'''

NOTE = '''A new browser game starts. Look at the screen and the brief, then write a short game note, exactly these
five lines:
Type: <the kind of game, e.g. shooter, platformer, puzzle, racing, idle, card>
Goal: <what you must do to win or do well>
Controls: <the keys and mouse actions that matter, and what each does>
Progress display: <where on screen the score, lives, level, money or timer show, on the 0-999 grid, or none>
What counts: <what to maximize: survival, points, time, distance, levels, money, or what else>'''


def parse(reply, width, height):
    """Gemma's reply -> {'controller': code}, an action dict in pixels or {'keep': True}. Raises ValueError if none."""
    code = re.search(r'```python\n(.*?)(```|$)', reply, re.S)
    if code:
        return {'controller': code.group(1)}
    start = reply.find('{')
    if start < 0:
        if re.search(r'\bKEEP\b', reply):
            return {'keep': True}
        raise ValueError('reply had no JSON action, python block or KEEP')
    choice = json.JSONDecoder().raw_decode(reply[start:])[0]
    return {'tool': choice['tool'], 'params': to_pixels(choice.get('params') or {}, width, height)}


def chat(messages, max_tokens):
    """One reply from the vLLM server: (text, tokens generated)."""
    body = json.dumps({'model': 'gemma', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}).encode()
    request = urllib.request.Request(SERVER, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=85) as response:
        out = json.load(response)
    return out['choices'][0]['message']['content'], out['usage']['completion_tokens']


class Agent(AutoPilot):
    replies = []   # every reply of every game, for reading after a run

    def ask(self, obs, event, system, text, max_tokens):
        frame = Image.open(io.BytesIO(obs.image)).convert('RGB')
        frame.thumbnail((768, 768))
        jpeg = io.BytesIO()
        frame.save(jpeg, 'JPEG', quality=90)
        image = 'data:image/jpeg;base64,' + base64.b64encode(jpeg.getvalue()).decode()
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}}, {'type': 'text', 'text': text}]},
        ]
        began = time.time()
        reply, tokens = chat(messages, max_tokens)
        self.replies.append((obs.elapsed_s, event, reply, time.time() - began, tokens))
        return reply

    def think(self, obs, event, memory):
        info = obs.info
        brief = f"Brief: {info['description']} Controls: {info['controls']}\n"
        if 'note' not in memory:   # once per game, before its first move
            memory['note'] = self.ask(obs, 'game note', NOTE, brief, 200).strip()
        log = memory.setdefault('log', [])
        progress = memory.setdefault('progress', [])
        text = (
            brief + f"Your game note:\n{memory['note']}\n"
            f"Time left: {info['remaining_s']:.0f} s.\n"
            f"What just happened: {event}.\n"
            f"Result of the last action: {obs.last_action}\n"
            "Progress you read so far (game time: progress): "
            + (', '.join(f'{t:.0f} s: {p}' for t, p in progress[-10:]) or 'none') + '\n'
            + (f"Your controller is playing now, for {obs.elapsed_s - self.controller_start:.0f} s. Its code:\n"
               f"```python\n{self.controller_code}```\n" if self.controller else '')
            + "Your recent replies, oldest first:\n" + ('\n'.join(log[-6:]) or 'none') + '\nYour reply:'
        )
        reply = self.ask(obs, event, RULES, text, 768)
        log.append(f'[{event}] ' + ' '.join(reply.split())[:300])
        found = re.search(r'^\s*Progress:\s*(.+)$', reply, re.M)
        if found:
            progress.append((obs.elapsed_s, found.group(1).strip()[:40]))
        return parse(reply, info['width'], info['height'])
