# Notebook cell: Gemma, served by vLLM on localhost (setup cell), in every role of autopilot.Agent.
import base64
import io
import json
import re
import threading
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

PLAYER = f'''You play a real-time browser game from screenshots, as the fast player of a team. The game keeps
running while you think: answer quickly and keep the game moving.

{TOOLS}

Your reply:
Progress: <what counts in this game (game note), as it stands on this screen, e.g. "score 120",
  "alive, 3 lives", "level 2", "timer 1:20", or "unknown">
Note: <only if a line of the game note is wrong: the corrected line, e.g. "Progress display: 0,930,999,999">
Then exactly one of:
- one action as JSON on one line, from the tools above
- KEEP             while a controller plays and works
- REUSE #<id>      restart a controller from the library
- WRITE: <goal>    a writer writes a controller for this goal (what it should find on screen and what it
                   should do) in about 10 s while you keep playing; it starts by itself when ready.
                   Ask for one when the game needs fast reactions (dodging, aiming, steering, timing).
Prefer a sequence for fixed routes and menu paths. If a menu or game-over screen shows, start the game the
way the brief or the screen says. If something did not help, try something different. Never repeat what
the library lists as failed.'''

CONTROLLER = '''A controller is Python code that plays every 200 ms from the screen:
```python
# skill: <one line: what this controller does>
def control(frame, mem):
    # frame: numpy uint8 array (1000, 1000, 3), RGB: the screen scaled to the 0-999 grid, so frame[y, x]
    #        is grid point (x, y). np.argwhere returns (y, x). np is numpy.
    # mem: a dict kept between calls.
    # Return one action dict from the tools, in grid coordinates as they come from frame,
    # or set mem["handoff"] = "<why>" and return None to give control back.
    ...
```
Rules:
- The first line is "# skill: <one line>"; it is the only comment. No other comments, docstrings or blank lines.
- Use the coordinates from frame as they are: frame already is the grid, never rescale them.
- Act on what you detect: aim at the targets you found, move away from the threats you found.
- Serve what counts in this game (the game note): survive, score, go fast or get far.
- Fast, well under 0.1 s per call: numpy color masks, never Python loops over pixels.
- Give control back when something happens it cannot handle.
Reply with one python block only.'''

WRITER = f'''You write controllers for a real-time browser game, for a teammate who plays meanwhile.

{TOOLS}

{CONTROLLER}'''

VARY = '''Below is a controller that works. Write a variant that should do clearly better at what counts, e.g.
the same plus dodging, better aim, or a control it ignores.'''

CHECKER = '''You check a controller that plays a real-time browser game. The images show the last seconds, oldest
first; the last image is now. Reply in exactly three lines:
Progress: <the number for what counts (game note) on the last image, or unknown>
Verdict: KEEP or REPLACE
Reason: <one line: what the controller does well or wrong, and what a better one should do>
REPLACE only when it clearly fails at what counts; while it works, KEEP.'''

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

PROBE = '''The game is running. Plan a quick control check: one test action for each control the brief or the
note names (each key, each mouse action), so you see what each one does. Skip any that restart, pause,
quit, or open settings or menus. At most 8 lines, each: <control>: <one JSON action from the tools>
Example:
Q (build): {"tool":"press_key","params":{"key":"q"}}
left click (dig): {"tool":"click","params":{"x":500,"y":500}}'''

LESSON = '''A run of a real-time browser game just ended (or a level changed). The images show the last seconds
before it, oldest first; the last image is now. Reply in one line:
Lesson: <what ended the run, and what to do differently next time>'''


def parse(reply, width, height):
    """The player's reply -> a decision for autopilot.Agent (actions in pixels). Raises ValueError if none."""
    code = re.search(r'```python\n(.*?)(```|$)', reply, re.S)
    if code:
        return {'controller': code.group(1)}
    write = re.search(r'^\s*WRITE:\s*(.+)$', reply, re.M)
    if write:
        return {'write': write.group(1).strip()}
    start = reply.find('{')
    if start < 0:
        reuse = re.search(r'\bREUSE\s*#?(\d+)', reply)
        if reuse:
            return {'reuse': int(reuse.group(1))}
        if re.search(r'\bKEEP\b', reply):
            return {'keep': True}
        raise ValueError('reply had no JSON action, KEEP, REUSE or WRITE')
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


def first_number(text):
    found = re.search(r'-?\d[\d,]*\.?\d*', text or '')
    return float(found.group(0).replace(',', '')) if found else None


def python_block(reply):
    code = re.search(r'```python\n(.*?)(```|$)', reply, re.S)
    return code and code.group(1)


def chat(messages, max_tokens, priority=0):
    """One reply from the vLLM server: (text, tokens generated). Lower priority is served first."""
    body = json.dumps({'model': 'gemma', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
                       'priority': priority}).encode()
    request = urllib.request.Request(SERVER, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=85) as response:
        out = json.load(response)
    return out['choices'][0]['message']['content'], out['usage']['completion_tokens']


class Agent(AutoPilot):
    replies = []   # every call of every game: (game time, role: event, reply, seconds, tokens, role, calls at once)
    in_flight = 0
    lock = threading.Lock()

    def ask(self, obs, role, event, system, text, max_tokens, images=()):
        content = []
        for jpeg in [*images, obs.image]:
            frame = Image.open(io.BytesIO(jpeg)).convert('RGB')
            frame.thumbnail((768, 768))
            out = io.BytesIO()
            frame.save(out, 'JPEG', quality=90)
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'
                                                                      + base64.b64encode(out.getvalue()).decode()}})
        content.append({'type': 'text', 'text': text})
        with Agent.lock:
            Agent.in_flight += 1
            together = Agent.in_flight
        began = time.time()
        try:
            reply, tokens = chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                                 max_tokens, priority=0 if role == 'player' else 1)   # the player goes first
        finally:
            with Agent.lock:
                Agent.in_flight -= 1
        Agent.replies.append((obs.elapsed_s, f'{role}: {event}', reply, time.time() - began, tokens, role, together))
        return reply

    @staticmethod
    def brief(obs):
        return f"Brief: {obs.info['description']} Controls: {obs.info['controls']}\n"

    @staticmethod
    def note_text(view):
        if view['note'] is None:
            return 'not written yet'
        return view['note'] + ''.join(f'\nCorrection: {fix}' for fix in view['fixes'][-3:])

    def context(self, obs, view):
        return (self.brief(obs) + f"Game note:\n{self.note_text(view)}\n"
                + ('Control check, what each control changed on screen:\n' + '\n'.join(view['probes']) + '\n'
                   if view['probes'] else '')
                + ('Lessons from earlier runs: ' + ' | '.join(view['lessons'][-3:]) + '\n' if view['lessons'] else '')
                + f"Controller library:\n{view['library']}\n")

    # --- the roles ---

    def think(self, obs, event, view):
        playing = view['playing']
        text = (self.context(obs, view)
                + f"Time left: {obs.info['remaining_s']:.0f} s.\n"
                f"What just happened: {event}.\n"
                f"Result of the last action: {obs.last_action}\n"
                f"Measured on screen: {view['signals']}\n"
                "Progress you read so far (game time: progress): "
                + (', '.join(f'{pt:.0f} s: {p}' for pt, p in view['progress'][-10:]) or 'none') + '\n'
                + (f"Playing now: controller #{playing['id']} ({playing['desc']}), for {playing['seconds']:.0f} s.\n"
                   if playing else 'No controller is playing.\n')
                + (f"The writer is writing: {view['writing']}\n" if view['writing'] else '')
                + "Your recent replies, oldest first:\n" + ('\n'.join(view['log'][-6:]) or 'none') + '\nYour reply:')
        reply = self.ask(obs, 'player', event, PLAYER, text, 200)
        progress = re.search(r'^\s*Progress:\s*(.+)$', reply, re.M)
        fix = re.search(r'^\s*Note:\s*(.+)$', reply, re.M)
        decision = parse(reply, obs.info['width'], obs.info['height'])
        decision['updates'] = {
            'log': f'[{event[:60]}] ' + ' '.join(reply.split())[:200],
            'progress': progress and (progress.group(1).strip()[:40], first_number(progress.group(1))),
            'note_fix': fix and fix.group(1).strip()[:200],
            'display_box': fix and display_box(fix.group(0)),
        }
        return decision

    def write(self, obs, goal, view):
        playing = view['playing']
        text = (self.context(obs, view)
                + (f"The controller playing now, #{playing['id']}:\n```python\n{playing['code']}```\n"
                   if playing and goal.startswith('replace') else '')
                + f"Write a controller for this goal: {goal}")
        return python_block(self.ask(obs, 'writer', goal[:60], WRITER, text, 1500))

    def vary(self, obs, entry, view):
        text = (self.context(obs, view) + f"Controller #{entry['id']} ({entry['desc']}):\n"
                f"```python\n{entry['code']}```\n{VARY}")
        return python_block(self.ask(obs, 'variant', f"of #{entry['id']}", WRITER, text, 1500))

    def check(self, obs, view):
        playing = view['playing']
        if not playing:
            return None
        read = ', '.join(f'{v:g}' for _, v in playing['readings'][-5:]) or 'none yet'
        text = (self.context(obs, view)
                + f"Controller #{playing['id']} ({playing['desc']}) has played for {playing['seconds']:.0f} s. "
                f"Numbers read so far: {read}. Measured on screen: {view['signals']}\n"
                f"```python\n{playing['code']}```")
        reply = self.ask(obs, 'checker', f"#{playing['id']}", CHECKER, text, 120, view['frames'][-6:-1])
        verdict = re.search(r'Verdict:\s*(KEEP|REPLACE)', reply)
        reason = re.search(r'^\s*Reason:\s*(.+)$', reply, re.M)
        progress = re.search(r'^\s*Progress:\s*(.+)$', reply, re.M)
        return {'verdict': verdict.group(1) if verdict else 'KEEP', 'reason': reason.group(1).strip() if reason else '',
                'reading': first_number(progress.group(1)) if progress else None}

    def analyse(self, obs, task, view):
        width, height = obs.info['width'], obs.info['height']
        if task == 'note':
            revise = view['note'] is not None
            reply = self.ask(obs, 'analyst', 'note revised' if revise else 'note', REVISE if revise else NOTE,
                             self.brief(obs) + ('Your note so far:\n' + self.note_text(view) if revise else ''), 250)
            return {'note': reply.strip(), 'display_box': display_box(reply)}
        if task == 'probe':
            reply = self.ask(obs, 'analyst', 'control check', PROBE + '\n\n' + TOOLS,
                             self.brief(obs) + 'Game note:\n' + self.note_text(view), 300)
            return {'probe': parse_probe(reply, width, height)}
        if task == 'lesson':
            reply = self.ask(obs, 'analyst', 'lesson', LESSON, self.context(obs, view), 150, view['frames'][-6:-1])
            lesson = re.search(r'Lesson:\s*(.+)', reply)
            return {'lesson': (lesson.group(1) if lesson else reply).strip()[:200]}
        return None
