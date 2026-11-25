# Generalist Gaming Agent – Projektplan

Kaggle-Wettbewerb: ein Offline-Agent spielt 32 unbekannte Browser-Spiele in Echtzeit (je 10 min Spielzeit).
Vorlage: AutoPILOT (arXiv:2611.28202) – das Modell startet neben Einzelaktionen auch **Macros** und
**Controller** (vom Modell live geschriebener Python-Code, der selbst den Bildschirm liest).

## Rahmenbedingungen (aus Overview/Data-Tab)

- Abgabe: Notebook schreibt `/kaggle/working/agent.py` mit `class Agent` (`__init__`, `step(obs, tools) -> Action`),
  letzte Zelle startet `GGAInferenceServer().run()`. Kein Internet. RTX Pro 6000 (96 GB). ≤ 9 h Laufzeit.
- Startup (Modellserver + `Agent.__init__`) ≤ 15 min. Jede Entscheidung ≤ 90 s, sonst endet das Spiel.
- Spiel läuft mit 20 fps weiter, Entscheidungspunkt alle 200 ms. Antwort wirkt am nächsten Entscheidungspunkt.
  Jede Antwort außer `skip` ersetzt laufende Eingaben.
- Beobachtung: `obs.image` (JPEG, Pixel = Klickkoordinaten), `obs.frames`, `obs.history`, `obs.info`
  (Brief: description, controls, remaining_s, width, height), `obs.last_action`, `obs.elapsed_s`.
- 11 Tools: `click, move, drag, scroll, press_key, hold_keys, key_down, key_up, wait, sequence, skip`.
- Neues Spiel erkennbar daran, dass `obs.elapsed_s` auf ~0 zurückfällt → Spielzustand zurücksetzen.
- Bewertung: pro Spiel 0 (nichts tun) bis 1 (Mensch), gemittelt. Private LB = 16 versteckte Spiele.

## Übertragung von AutoPILOT auf diese Schnittstelle

| AutoPILOT | Hier |
|-----------|------|
| `act` | ein normales Tool (`click`, `press_key`, …) |
| `run_macro` | **gibt es schon**: Tool `sequence` (≤ 200 Schritte, läuft während das Modell denkt) |
| `python`-Controller | **selbst bauen**, im Agent-Prozess (siehe unten) |
| Wake-Events | selbst berechnen in `step()` |

**Kernarchitektur: `step()` blockiert nie auf das LLM.**

```
step(obs) ──► Frame in Puffer legen, Wake-Events prüfen
         ├─ LLM-Thread fertig? → seine Entscheidung übernehmen (Action / sequence / neuer Controller)
         ├─ Controller aktiv?  → controller(obs, mem) → Action  (reines NumPy, wenige ms)
         ├─ LLM idle + Wake-Event? → LLM-Aufruf im Hintergrund-Thread starten
         └─ sonst → skip (laufende Eingaben/sequence weiterspielen lassen)
```

- Controller-Code: Modell schreibt `def control(frame, mem) -> dict | None` (None = Handoff ans Modell).
  Ausgeführt per `exec` mit NumPy; Zeitlimit pro Aufruf messen, bei Fehler/zu langsam → abschalten + Wake.
  (Kein Docker auf Kaggle; Sandbox nur leichtgewichtig – reicht, da nur unser eigenes Modell Code schreibt.)
- Wake-Events: Controller-Handoff oder -Fehler, `sequence` fertig, Szenenwechsel (Pixeldiff), Timeout,
  abgelehnte Aktion (`obs.last_action`), neues Spiel.
- Modell: Gemma 4 31B (4-bit) in plain transformers wie im Starter-Notebook, lädt in ~1,5 min.
- Budget: 32 × 10 min ≈ 5,3 h Spielzeit – passt in 9 h.

## Arbeitsweise

Gemma (62 GB) läuft nur auf Kaggle. Hier wird nur der modellfreie Teil getestet (`python test_agent.py`).
`python make_notebook.py` baut aus den Repo-Dateien `gga_autopilot.ipynb`; das wird in Kaggle importiert
(File → Import Notebook), ausgeführt, und `replies.txt` + die Ausgabe der Play-Zelle kommen zurück.

## Schritte

1. **Daten + lokale Schleife**: Wettbewerbs-Dataset holen, `gga/protocol.py` lesen,
   `python -m gga.game --game off_line --agent agent.py` mit Dummy-Agent laufen lassen.
2. ✅ **Async-Gerüst ohne LLM** (`autopilot.py`, Test: `python test_agent.py`): `Agent` mit Hintergrund-Thread, Frame-Puffer, Spielwechsel-Erkennung,
   Wake-Events. Fake-„LLM“, das nach 3 s eine feste Aktion liefert → prüft, dass `step()` nie blockiert.
3. 🧪 **Gemma-Anbindung** (`gemma.py`, Notebook: `python make_notebook.py` → `gga_autopilot.ipynb`, in Kaggle testen). Ursprünglich: **Baseline-LLM**: Starter-Notebook-Modell anbinden, eine Aktion pro Aufruf (Messlatte auf den 4 Beispielspielen).
4. **`sequence` nutzen**: Modell darf Aktionsfolgen ausgeben.
5. **Controller**: Modell schreibt `control()`, Ausführung pro `step()`, Ersetzen/Stoppen, Handoff.
6. **Prompt**: Regeln aus dem Paper („Prefer sequence when it is enough“, „Test an uncertain control
   briefly before you rely on it“), Notizen pro Spiel.
7. **Kaggle-Notebook**: Abgabe, 4 Beispielspiele lokal messen, dann submitten (~6 h bis Score).
