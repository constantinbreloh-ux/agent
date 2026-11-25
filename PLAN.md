# Generalist Gaming Agent – Projektplan

Ziel: Agent für den Kaggle-Wettbewerb „Generalist Gaming Agent“, der unbekannte Browser-Spiele spielt.
Vorlage: AutoPILOT (arXiv:2611.28202), ein Harness, in dem das Modell neben Einzelaktionen auch
**Macros** (offene Aktionsfolgen) und **Controller** (Python-Schleifen, die selbst den Bildschirm lesen) startet.

## Kernidee aus dem Paper

Ein Modellaufruf = ein Tool-Call, aber der Tool-Call kann eines von drei Dingen starten:

| Tool | Was passiert | Wann |
|------|--------------|------|
| `act(action)` | eine Aktion; stoppt laufendes Programm | Menüs, Tests, Einzelschritte |
| `run_macro([actions])` | Liste von Aktionen, ohne Bild zu lesen (max. 200) | feste Routen, Menüfolgen |
| `python(code)` | `def run(api)` läuft im Hintergrund, ~alle 0,4 s `api.observe()` → `api.act()` | Echtzeit: Ausweichen, Zielen |

Während Macro/Controller laufen, wartet das Modell (keine Tokens). Es wird geweckt durch **Wake-Events**:
- vom Programm: `api.request_handoff(reason)`, `run()` endet/Fehler, Macro fertig
- vom Harness: Szenenwechsel (großer Teil des Bildes anders und stabil), Entscheidungsbildschirm, Timeout (120 s)

Controller-API: `api.observe()` (RGB-Frame 720×1280×3 + Spieluhr/Budgets/`reset_epoch`),
`api.act({...})`, `api.request_handoff(reason)`, `api.notify_reset(reason)`.
Aktionsraum: `wait, press_key, press_keys, type, click, click_hold, mouse_move, drag, scroll`;
Koordinaten auf 1280×720, Dauer ≤ 2 s (Standard 0,4 s).

Ergebnis im Paper: Echtzeit-Spiele 68,6 % statt 38,3 % ohne Macros/Controller; 47 % weniger Tokens.

## Schritte

Jeder Schritt ist für sich lauffähig und testbar.

0. **Wettbewerbsregeln klären** (offen, siehe unten): Spiele-Schnittstelle, Abgabeformat, Zeit-/GPU-Limits,
   ob Code-Ausführung im Notebook erlaubt ist.
1. **Lokale Spielumgebung**: Playwright-Browser mit einem einfachen HTML-Spiel; `Env` mit
   `observe() -> np.ndarray` und `act(action: dict)`. Später durch die echte Kaggle-Schnittstelle ersetzt.
2. **Baseline-Agent**: Gemma-4-31B (aus dem Starter-Notebook) wählt pro Aufruf genau eine Aktion
   (`act`). Das ist die Messlatte.
3. **`run_macro`**: Aktionsliste ausführen, am Ende Wake-Event „Macro fertig“.
4. **`python`-Controller**: Modell-Code in einem Subprozess (Timeout, kein Netzwerk) mit der `api` oben;
   ersetzen/stoppen bei neuem `python`-Call oder `act`.
5. **Wake-Events**: Szenenwechsel per Pixeldifferenz, Timeout, Handoff. Polling alle 0,25 s.
6. **Prompt**: Regeln aus dem Paper („Prefer `run_macro` when it is enough“, „Test an uncertain control
   briefly before you rely on it“), Notizdatei pro Spiel.
7. **Kaggle-Paket**: alles in das Abgabeformat bringen, auf echten Spielen messen.

## Offene Fragen (blockieren Schritt 0)

- Wie werden die Spiele bereitgestellt (eigene Env-API, URL, Screenshots)? Welches Aktionsformat?
- Abgabeformat: Notebook, Agent-Klasse, bestimmte Funktionssignatur?
- Limits: GPU, Zeit pro Schritt/Spiel, Internet aus?
- Läuft das Spiel in Echtzeit weiter, während das Modell denkt? (Entscheidet, wie viel Controller bringt.)
