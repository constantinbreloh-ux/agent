# Plays the sample games here, one Agent for all as in the competition, and prints a measurement row per game.
import statistics
from collections import Counter
from IPython.display import HTML, Image as Frame, display
from gga.game import Game

GAMES = ['asteroid_capitalist', 'blarglike', 'cavebots', 'off_line']
PLAY_SECONDS = 120             # per game; in the competition a game runs for 10 minutes

# The competition's run skips this: the server in the last cell must be up within 15 minutes.
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    specs = json.load(open(harness / 'games.json'))['games']
    agent = Agent()
    Agent.replies.clear()
    screen = display(HTML('<i>Starting...</i>'), display_id=True)
    status = display(HTML(''), display_id=True)
    sheet = []
    log = open('/kaggle/working/replies.txt', 'w')   # every Gemma reply, to read after the run
    for name in GAMES:
        game = Game(name, video_dir=f'/kaggle/working/video_{name}')
        game.initialize({'minutes': PLAY_SECONDS / 60})
        tools = game.tools()
        first_reply = len(Agent.replies)
        turns, refused, step_s, tools_used = 0, [], [], Counter()
        while not game.is_terminal():
            obs = game.get_observation()
            began = time.time()
            action = agent.step(obs, tools)
            step_s.append(time.time() - began)
            result = game.handle_tool(action.tool, action.params)
            turns += 1
            tools_used[('controller: ' if agent.controller else '') + action.tool] += 1
            if not result['ok'] and not game.is_terminal():
                refused.append(result.get('error'))
            if turns % 5 == 0:
                screen.update(Frame(data=obs.image, width=640))
                status.update(HTML(f'{name}: {obs.elapsed_s:.0f} s, controller {"on" if agent.controller else "off"}, '
                                   f'last event: <code>{agent.events[-1][1][:120] if agent.events else ""}</code>'))
        game.stop()
        score = game.get_score()   # local runs only: the game's own counter, normalized 0..1 like the leaderboard
        replies = Agent.replies[first_reply:]
        log.write(f'##### {name}\n')   # written first: an error in the measurements below must not lose it
        for elapsed, event, reply, secs, tokens, role, together in replies:
            log.write(f'=== {elapsed:.1f} s, {event} ({secs:.1f} s, {tokens} tokens, {together} at once)\n{reply}\n\n')
        log.flush()
        print('Events:', [(round(t, 1), e[:100]) for t, e in agent.events])
        try:
            writes = [r for r in replies if r[5] in ('writer', 'variant')]
            def median(values):
                values = list(values)
                return round(statistics.median(values), 1) if values else None
            roles = {role: f"{sum(r[5] == role for r in replies)} calls, {median(r[3] for r in replies if r[5] == role)} s, "
                           f"{median(r[4] for r in replies if r[5] == role)} tok"
                     for role in ('player', 'writer', 'variant', 'checker', 'analyst')}
            alone = median(r[3] for r in replies if r[5] == 'player' and r[6] == 1)
            shared = median(r[3] for r in replies if r[5] == 'player' and r[6] > 1)
            stuck = [(round(t), e[:60]) for t, e in agent.events if e.startswith('no progress')]
            row = {
                'game': name, 'metric': specs[name]['scoring']['metric'], 'score': score,
                'skip share': round(tools_used['skip'] / max(turns, 1), 2), 'refused': len(refused), 'refusals': refused[:3],
                'step max ms': round(max(step_s) * 1000),
                'replies': len(replies), 'reply s median': round(statistics.median(r[3] for r in replies), 1) if replies else None,
                'tokens median': statistics.median(r[4] for r in replies) if replies else None,
                'controller writes/min': round(len(writes) / (PLAY_SECONDS / 60), 1),
                'attempts': len(agent.attempts) + 1, 'attempt s': [round(a) for a in agent.attempts],
                'restart → controller s': agent.restart_delays, 'no-progress wakes': stuck,
                'library (id, what, status, signals/min, read progress/min)': [
                    (e['id'], e['desc'][:40], e['status'], round(agent.rate(e, obs.elapsed_s)), round(agent.gain(e)))
                    for e in agent.library],
                'roles': roles, 'player s alone / with others': (alone, shared),
                'skip share while writing': round(agent.stats['skips while writing'] / agent.stats['steps while writing'], 2)
                if agent.stats['steps while writing'] else None,
                'first scene change → first action s': agent.first_play_delay,
                'checker verdicts': dict(agent.verdicts),
                'variants': dict(agent.variants), 'display box': agent.memory.get('display_box'),
                'display ticks like a clock': agent.ticking,
                'probes': agent.memory.get('probes'), 'note': agent.memory.get('note'),
                'lessons': agent.memory.get('lessons'), 'actions': dict(tools_used),
            }
        except Exception as e:   # one game's measurements failing must not end the run
            import traceback
            row = {'game': name, 'error': traceback.format_exc()}
        sheet.append(row)
        print(json.dumps(row, indent=1, ensure_ascii=False, default=str))
    log.close()
    json.dump(sheet, open('/kaggle/working/sheet.json', 'w'), indent=1, ensure_ascii=False, default=str)
    print('\nMeasurement sheet')
    for row in sheet:
        if 'error' in row:
            print(f"{row['game']:20} measurement error: {row['error'][-300:]}")
            continue
        print(f"{row['game']:20} score {row['score']}  skip {row['skip share']}  refused {row['refused']}  "
              f"reply {row['reply s median']} s / {row['tokens median']} tok  writes/min {row['controller writes/min']}  "
              f"attempts {row['attempts']} {row['attempt s']}  no-progress {len(row['no-progress wakes'])}  "
              f"restart {row['restart → controller s']}  variants {row['variants']}  clock {row['display ticks like a clock']}\n"
              f"{'':20} player s alone/shared {row['player s alone / with others']}  skip while writing "
              f"{row['skip share while writing']}  first action {row['first scene change → first action s']} s  "
              f"verdicts {row['checker verdicts']}\n{'':20} {row['roles']}")
