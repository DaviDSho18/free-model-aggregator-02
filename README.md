# free-model-aggregator-02 — quality-ranked free-model aggregator

One server, two client wires (OpenAI Chat Completions and Anthropic Messages), and a real free
OpenRouter model behind every answer, dialled through the grader's gateway.

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/DaviDSho18/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/DaviDSho18/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Built by an AI agent for [make.fun](https://make.fun)'s "The best live free-model aggregator" bounty.

## What it does

- Reads the trial brief, ranks the free models that meet each prompt's requirements by the brief's own
  published rating, and serves every prompt from the best one that answers.
- Retries a refused prompt on the next-ranked model and reports each recovered refusal in the
  `x-makefun-upstream-failures` header.
- Cools a model down after it refuses, counted in calls rather than wall-clock time, so one
  rate-limited model cannot take every later prompt down with it and a replay still reproduces.
- Registers the exact fourteen dialable free ids and answers `GET /v1/models` with that same set on
  both auth headers.
- Polls `GET $MAKEFUN_ENV_URL/work` and exits 0 the moment it says done.

## Run

```sh
node main.ts --list      # print the catalog and exit
node main.ts             # serve (needs MAKEFUN_ENV_URL and OPENAI_BASE_URL)
```

Node 22 or later. No packages: the standard library alone.

## Layout

| Path | What it holds |
| --- | --- |
| `main.ts` | the server: routes, register, poll |
| `lib/providers.ts` | the brief's types and parser, the free model list, the catalog |
| `lib/wire.ts` | Anthropic ↔ Chat Completions translation, the model field, the SSE writers |
| `lib/routing.ts` | quality-first candidate ranking, per-model cooldowns, the refusal trail |

## License

MIT — see `LICENSE`.
