# Notebook cell: Gemma as the model behind autopilot.Agent.
# `processor` and `model` come from the setup cell; they are only used inside think().
import io
import json
import re

from PIL import Image

from autopilot import Agent as AutoPilot

RULES = '''You play a real-time browser game from screenshots. The game keeps running while you think, and each
of your replies takes seconds, so prefer moves that keep playing on their own.

Reply in exactly one of these two forms.

1. One action as JSON on one line. Coordinates are on a 0-999 grid over the image (x: left to right,
   y: top to bottom):
{"tool":"click","params":{"x":500,"y":500}}
{"tool":"move","params":{"x":500,"y":500}}
{"tool":"press_key","params":{"key":"Space"}}
{"tool":"sequence","params":{"steps":[{"keys":["ArrowRight"],"buttons":[],"x":500,"y":500,"duration_ms":400},{"keys":["ArrowUp","x"],"buttons":["left"],"x":500,"y":500,"duration_ms":200}]}}
   A sequence plays its steps in order, each holding its keys and mouse buttons for duration_ms
   (0 = hold until your next reply). Use it for fixed routes, menu paths and timed combos.

2. A controller: Python code that plays every 200 ms from the screen without you. Use it when the game
   needs fast reactions (dodging, aiming, steering, timing jumps).
```python
def control(frame, mem):
    # frame: numpy uint8 array (H, W, 3), RGB, the full game screen; W and H are given below. np is numpy.
    # mem: a dict kept between calls.
    # Return one action dict in PIXEL coordinates (not 0-999), same tools as above, e.g.
    #   {"tool": "press_key", "params": {"key": "ArrowLeft"}}
    # or set mem["handoff"] = "<why>" and return None to give control back to you.
    ...
```
   Keep control() fast, well under 0.1 s: find things by color with numpy masks, never loop over pixels.
   Give control back when something happens it cannot handle.

Prefer a sequence when it is enough. Test an uncertain control briefly with one action before you rely on it.
If a menu or game-over screen shows, start the game the way the brief or the screen says.
If something you tried did not help, try something different.'''


def to_pixels(params, width, height):
    for target in [params, *params.get('steps', [])]:
        if 'x' in target:
            target['x'] = min(max(int(target['x']), 0), 999) * (width - 1) // 999
            target['y'] = min(max(int(target['y']), 0), 999) * (height - 1) // 999
    return params


def parse(reply, width, height):
    """Gemma's reply -> {'controller': code} or an action dict in pixels. Raises ValueError if neither."""
    code = re.search(r'```python\n(.*?)(```|$)', reply, re.S)
    if code:
        return {'controller': code.group(1)}
    start = reply.find('{')
    if start < 0:
        raise ValueError('reply had neither a JSON action nor a python block')
    choice = json.JSONDecoder().raw_decode(reply[start:])[0]
    return {'tool': choice['tool'], 'params': to_pixels(choice.get('params', {}), width, height)}


class Agent(AutoPilot):
    replies = []   # every reply of every game, for reading after a run

    def think(self, obs, event, memory):
        info = obs.info
        log = memory.setdefault('log', [])
        frame = Image.open(io.BytesIO(obs.image)).convert('RGB')
        frame.thumbnail((768, 768))
        text = (
            f"Brief: {info['description']} Controls: {info['controls']}\n"
            f"Screen: {info['width']}x{info['height']} pixels. Time left: {info['remaining_s']:.0f} s.\n"
            f"What just happened: {event}.\n"
            f"Result of the last action: {obs.last_action}\n"
            f"Your recent replies, oldest first:\n" + ('\n'.join(log[-6:]) or 'none') + '\nYour reply:'
        )
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': RULES}]},
            {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # noqa: F821
        inputs = processor(text=prompt, images=[frame], return_tensors='pt').to(model.device)  # noqa: F821
        out = model.generate(**inputs, max_new_tokens=768, do_sample=False, prompt_lookup_num_tokens=20)  # noqa: F821
        reply = processor.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)  # noqa: F821
        log.append(f'[{event}] ' + ' '.join(reply.split())[:300])
        self.replies.append((obs.elapsed_s, event, reply))
        return parse(reply, info['width'], info['height'])
