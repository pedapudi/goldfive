# harmonograf_observed example

Minimal `CallableAdapter` agent wired to both an `InMemorySink` and a
`HarmonografSink` so every goldfive event lands in the harmonograf UI.
No LLM, no ADK, no Claude SDK — just a four-task `StaticPlanner` and a
canned-reply callable.

## What this shows

- A single goldfive `Runner` fanning events out to two sinks.
- The standard `Client` / `HarmonografSink` wiring from
  `harmonograf-client` and the paired `runner.close()` +
  `client.shutdown(flush_timeout=5.0)` teardown.
- Graceful fallback: if `harmonograf_client` is not installed the
  example still runs end-to-end with `InMemorySink` + `LoggingSink` and
  prints a pointer to the observability guide.
- The event counts reported by `InMemorySink` match what the harmonograf
  console renders — useful as a sanity check when debugging the sink.

## Prerequisites

A harmonograf server reachable at `127.0.0.1:7531` (override with
`HARMONOGRAF_SERVER=host:port`). See the harmonograf operator
quickstart for `make demo`. Without a server running the example falls
back to the in-process sinks and still completes — the UI just stays
empty.

## How to run

With harmonograf installed and a server up:

```bash
uv pip install harmonograf-client
uv run python examples/harmonograf_observed/agent.py
```

Without harmonograf (fallback path, zero extra deps):

```bash
uv run python examples/harmonograf_observed/agent.py
```

Point at a non-default server:

```bash
HARMONOGRAF_SERVER=10.0.0.5:7531 \
  uv run python examples/harmonograf_observed/agent.py
```

## What to look for in the UI

- The plan materialising with four tasks (`research`, `draft`,
  `review`, `publish`) and three edges between them.
- A `TaskStarted` / `TaskCompleted` pair per task as the sequential
  executor walks the DAG.
- A `RunCompleted` event closing the stream — the session row should
  flip to green.
- Matching event sequence numbers between the UI and the
  `InMemorySink captured N events.` line printed at the end.

## See also

- [docs/guides/observability-with-harmonograf.md](../../docs/guides/observability-with-harmonograf.md)
  — full walkthrough covering plan diffs, drift markers, and control
  actions from the UI.
- [examples/hello_callable.py](../hello_callable.py) — the same shape
  without the harmonograf dependency, useful as a zero-deps starting
  point.
