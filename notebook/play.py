# Plays one sample game here, live, and shows what the agent does. Change the three settings and rerun this cell.
import queue
from collections import Counter
from IPython.display import HTML, Image as Frame, display
from gga.game import Game

GAME = 'asteroid_capitalist'   # asteroid_capitalist, blarglike, cavebots or off_line
PLAY_SECONDS = 60              # in the competition a game runs for 10 minutes
AGENT_VIEW = False             # False: the live game, every frame; True: only the screenshots the agent sees

# The competition's run skips this: the server in the last cell must be up within 15 minutes.
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    game = Game(GAME, video_dir='/kaggle/working')   # also saves a recording to the output files
    game.initialize({'minutes': PLAY_SECONDS / 60})
    tools = game.tools()
    agent = Agent()
    Agent.replies.clear()

    screen = display(HTML('<i>Starting...</i>'), display_id=True)
    status = display(HTML(''), display_id=True)
    live = queue.Queue()   # each turn: (when it came, its frames as (elapsed_s, jpeg)); None ends play_live

    def play_live():
        shown = last = None
        while (turn := live.get()) is not None:
            at, batch = turn
            step = (at - last) / len(batch) if last else 0.05
            last, start = at, time.time()
            for i, (elapsed, jpeg) in enumerate(batch, 1):
                if shown is None or elapsed > shown:
                    screen.update(Frame(data=jpeg, width=800))
                    shown = elapsed
                if not live.qsize():
                    time.sleep(max(0, start + i * step - time.time()))

    player = threading.Thread(target=play_live)
    player.start()

    turns, refused, step_s, tools_used = 0, [], [], Counter()
    while not game.is_terminal():
        obs = game.get_observation()
        if AGENT_VIEW:
            screen.update(Frame(data=obs.image, width=800))
        else:
            live.put((time.time(), obs.history + obs.frames))
        began = time.time()
        action = agent.step(obs, tools)
        step_s.append(time.time() - began)
        result = game.handle_tool(action.tool, action.params)
        turns += 1
        tools_used[('controller: ' if agent.controller else '') + action.tool] += 1
        if not result['ok'] and not game.is_terminal():
            refused.append(result.get('error'))
        last_event = agent.events[-1] if agent.events else ''
        status.update(HTML(f'Turn {turns}, game time {obs.elapsed_s:.0f} s, '
                           f'controller {"on" if agent.controller else "off"}, last event: <code>{last_event}</code>'))

    game.stop()
    live.put(None)
    player.join()
    print(f'{turns} turns, step() max {max(step_s) * 1000:.0f} ms, {len(Agent.replies)} Gemma replies, '
          f'{len(refused)} refused {refused[:3]}')
    print('Actions:', dict(tools_used))
    print('Events:', agent.events)
    with open('/kaggle/working/replies.txt', 'w') as f:   # every Gemma reply, to read after the run
        for elapsed, event, reply in Agent.replies:
            f.write(f'=== {elapsed:.1f} s, {event}\n{reply}\n\n')
    for elapsed, event, reply in Agent.replies[-3:]:
        print(f'\n=== {elapsed:.1f} s, {event}\n{reply}')
