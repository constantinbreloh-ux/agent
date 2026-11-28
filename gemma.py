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
If a line of your game note is wrong, add a line: Note: <the corrected line, e.g. "Progress display: 0,930,999,999">
After a game over, add a line: Lesson: <what ended the run and what you change>
Then reply in exactly one of these four forms.

1. One action, as JSON on one line, from the tools above.

2. A controller: Python code that plays every 200 ms from the screen without you. Use it when the game
   needs fast reactions (dodging, aiming, steering, timing jumps).
```python
# skill: <one line: what this controller does>
def control(frame, mem):
    # frame: numpy uint8 array (1000, 1000, 3), RGB: the screen scaled to the 0-999 grid, so frame[y, x]
    #        is grid point (x, y). np.argwhere returns (y, x). np is numpy.
    # mem: a dict kept between calls.
    # Return one action dict from the tools above, in grid coordinates as they come from frame,
    # or set mem["handoff"] = "<why>" and return None to give control back to you.
    ...
```
   Rules for controllers:
   - The first line is "# skill: <one line>"; it is the only comment. No other comments, docstrings or
     blank lines: every token you write costs game time.
   - Use the coordinates from frame as they are: frame already is the grid, never rescale them.
   - Act on what you detect: aim at the targets you found, move away from the threats you found.
     Code that finds things and then ignores them is useless.
   - Serve what counts in this game (your game note): survive, score, go fast or get far.
   - Fast, well under 0.1 s per call: numpy color masks, never Python loops over pixels.
   - Give control back when something happens it cannot handle.

3. KEEP
   Only while your controller is playing and it works: it plays on unchanged. When what counts is
   survival and you are still alive, that is progress.

4. REUSE #<id>
   Restart a controller from your library below.

Prefer a sequence when it is enough. Test an uncertain control briefly with one action before you rely on it.
If a menu or game-over screen shows, start the game the way the brief or the screen says.
If something you tried did not help, try something different. Never repeat what your library lists as failed.'''

NOTE = '''Look at the screen and the brief, then write a short game note, exactly these six lines:
Type: <the kind of game, e.g. shooter, platformer, puzzle, racing, strategy, idle, card>
Goal: <what you must do to win or do well>
Controls: <the keys and mouse actions that matter, and what each does>
Progress display: <x0,y0,x1,y1: the box on the 0-999 grid around the score, lives, level, money, timer or
  bar, usually at a screen edge; or none if this screen shows none>
What counts: <what to maximize: survival, points, time, distance, levels, money, or what else>
Strategy: <from what counts: for survival take few risks; for points try a lot; and your first plan>'''

REVISE = NOTE + '''
The game is running now. Your note so far is below; correct every line this screen shows better.'''

PROBE = '''The game is running. Plan a quick control check: one test action for each control the brief or your
note names (each key, each mouse action), so you see what each one does. Skip any that restart, pause,
quit, or open settings or menus. At most 8 lines, each: <control>: <one JSON action from the tools>
Example:
Q (build): {"tool":"press_key","params":{"key":"q"}}
left click (dig): {"tool":"click","params":{"x":500,"y":500}}'''

VARY = '''Below is one of your controllers that works. Write a variant that should do clearly better at what
counts, e.g. the same plus dodging, better aim, or using a control it ignores. Reply with one python
block only, first line "# skill: <one line>", same rules as before.'''


def parse(reply, width, height):
    """Gemma's reply -> a decision for autopilot.Agent (actions in pixels). Raises ValueError if none."""
    code = re.search(r'```python\n(.*?)(```|$)', reply, re.S)
    if code:
        return {'controller': code.group(1)}
    start = reply.find('{')
    if start < 0:
        reuse = re.search(r'\bREUSE\s*#?(\d+)', reply)
        if reuse:
            return {'reuse': int(reuse.group(1))}
        if re.search(r'\bKEEP\b', reply):
            return {'keep': True}
        raise ValueError('reply had no JSON action, python block, KEEP or REUSE')
    choice = json.JSONDecoder().raw_decode(reply[start:])[0]
    return {'tool': choice['tool'], 'params': to_pixels(choice.get('params') or {}, width, height)}


def parse_probe(reply, width, height):
    """'<label>: {json}' lines -> [(label, action in pixels)], watching the screen alone first."""
    probes = [('no input (the screen alone)', None)]
    for line in reply.splitlines():
        label, _, rest = line.partition(':')
        start = rest.find('{')
        if start < 0:
            continue
        try:
            action = json.JSONDecoder().raw_decode(rest[start:])[0]
            probes.append((label.strip()[:40], {'tool': action['tool'],
                                                'params': to_pixels(action.get('params') or {}, width, height)}))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return probes[:9]


def display_box(text):
    """The 'Progress display: x0,y0,x1,y1' box of a note, or None."""
    found = re.findall(r'Progress display:\s*\(?(\d+)\D+(\d+)\D+(\d+)\D+(\d+)', text)
    if not found:
        return None
    x0, y0, x1, y1 = (min(int(v), 999) for v in found[-1])   # the last one: a correction overrides the note
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def chat(messages, max_tokens):
    """One reply from the vLLM server: (text, tokens generated)."""
    body = json.dumps({'model': 'gemma', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}).encode()
    request = urllib.request.Request(SERVER, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=85) as response:
        out = json.load(response)
    return out['choices'][0]['message']['content'], out['usage']['completion_tokens']


class Agent(AutoPilot):
    replies = []   # every reply of every game, for reading after a run

    def ask(self, obs, event, system, text, max_tokens, images=()):
        content = []
        for jpeg in [*images, obs.image]:
            frame = Image.open(io.BytesIO(jpeg)).convert('RGB')
            frame.thumbnail((768, 768))
            out = io.BytesIO()
            frame.save(out, 'JPEG', quality=90)
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'
                                                                      + base64.b64encode(out.getvalue()).decode()}})
        content.append({'type': 'text', 'text': text})
        began = time.time()
        reply, tokens = chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], max_tokens)
        self.replies.append((obs.elapsed_s, event, reply, time.time() - began, tokens))
        return reply

    def library_text(self, memory, t):
        lines = []
        for e in memory['library']:
            if e['status'] in ('ok', 'running') or e['status'].startswith('worse'):
                lines.append(f"#{e['id']} [screen {e['scene']}] {e['desc']}: played {self.runtime(e, t):.0f} s, "
                             f"{self.rate(e, t):.0f} progress signals/min, {e['status']}")
            else:
                lines.append(f"#{e['id']} {e['desc']}: FAILED ({e['status']}: {e['reason'][:100]})")
        return '\n'.join(lines[-12:]) or 'empty'

    def note_text(self, memory):
        return memory['note'] + ''.join(f'\nCorrection: {fix}' for fix in memory.get('fixes', [])[-3:])

    def think(self, obs, event, memory):
        info, t = obs.info, obs.elapsed_s
        brief = f"Brief: {info['description']} Controls: {info['controls']}\n"
        if 'note' not in memory:   # at the very start, often a menu: rewritten once the game shows
            memory['note'] = self.ask(obs, 'game note', NOTE, brief, 250).strip()
        elif ((event.startswith('the scene changed') and not memory.get('note_after_scene'))
              or (t >= 20 and not memory.get('note_at_20s'))):
            memory['note_after_scene' if event.startswith('the scene changed') else 'note_at_20s'] = True
            memory['note'] = self.ask(obs, 'game note revised', REVISE,
                                      brief + 'Your note so far:\n' + self.note_text(memory), 250).strip()
            memory['fixes'] = []
        memory['display_box'] = display_box(self.note_text(memory))
        if not memory.get('probed') and (memory.get('note_after_scene') or memory.get('note_at_20s')):
            memory['probed'] = True   # once per game, as soon as the game itself runs
            probes = parse_probe(self.ask(obs, 'control check', PROBE + '\n\n' + TOOLS,
                                          brief + 'Your game note:\n' + self.note_text(memory), 300),
                                 info['width'], info['height'])
            if len(probes) > 1:
                return {'probe': probes}

        log = memory.setdefault('log', [])
        progress = memory.setdefault('progress', [])
        lessons = memory.setdefault('lessons', [])
        game_over = event.startswith('game over')
        text = (
            brief + f"Your game note:\n{self.note_text(memory)}\n"
            + ('Control check, what each control changed on screen:\n' + '\n'.join(memory['probes']) + '\n'
               if memory.get('probes') else '')
            + f"Time left: {info['remaining_s']:.0f} s.\n"
            f"What just happened: {event}.\n"
            f"Result of the last action: {obs.last_action}\n"
            f"Measured on screen: {memory.get('signals', 'nothing yet')}\n"
            + ('Lessons from earlier runs: ' + ' | '.join(lessons[-3:]) + '\n' if lessons else '')
            + "Progress you read so far (game time: progress): "
            + (', '.join(f'{pt:.0f} s: {p}' for pt, p in progress[-10:]) or 'none') + '\n'
            + f"Your controller library:\n{self.library_text(memory, t)}\n"
            + (f"Playing now: #{self.entry['id']}, for {t - self.controller_start:.0f} s. Its code:\n"
               f"```python\n{self.entry['code']}```\n" if self.controller and self.entry else '')
            + ('The first images show the last seconds before the game over, oldest first; the last image is now.\n'
               if game_over else '')
            + "Your recent replies, oldest first:\n" + ('\n'.join(log[-6:]) or 'none') + '\nYour reply:'
        )
        before = [jpeg for _, jpeg in list(memory['last_frames'])[-6:-1]] if game_over else []
        reply = self.ask(obs, event, RULES, text, 768, before)
        log.append(f'[{event[:60]}] ' + ' '.join(reply.split())[:300])
        for key, store in (('Progress', progress), ('Lesson', lessons), ('Note', memory.setdefault('fixes', []))):
            found = re.search(rf'^\s*{key}:\s*(.+)$', reply, re.M)
            if found:
                store.append((t, found.group(1).strip()[:40]) if key == 'Progress' else found.group(1).strip()[:200])
        return parse(reply, info['width'], info['height'])

    def vary(self, obs, entry, memory):
        info = obs.info
        text = (f"Brief: {info['description']} Controls: {info['controls']}\n"
                f"Your game note:\n{self.note_text(memory)}\n"
                f"Controller #{entry['id']} ({entry['desc']}), {self.rate(entry, obs.elapsed_s):.0f} progress signals/min:\n"
                f"```python\n{entry['code']}```\n{VARY}")
        code = re.search(r'```python\n(.*?)(```|$)', self.ask(obs, f"variant of #{entry['id']}", RULES, text, 768), re.S)
        return code and code.group(1)
