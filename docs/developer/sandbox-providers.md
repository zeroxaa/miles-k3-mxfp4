---
title: Sandbox Providers
description: How a cloud sandbox provider plugs into Miles, how to add one, and what each has been proven to do.
---

Connectors that run a task in a container (Harbor, OpenEnv) get that container
from a sandbox provider. A provider is **one entry in `PROVIDER_CREDENTIALS`**
(`miles/rollout/agentic/credentials.py`) — no connector code branches on it.
The in-process Harbor path passes `HARBOR_ENV_TYPE` straight to Harbor, and
both real-platform tests parametrize over that registry.

## Adding one

1. **The registry entry.** What credential a worker needs, the path-valued
   variable the launcher forwards in place of the secret, the address variables
   that are safe to forward by value, and the SDK. Field semantics are
   documented on `PROVIDER_CREDENTIALS` itself.
2. **A golden episode** on the real platform:
   `python scripts/sandbox_smoke/run.py --connector harbor --backend <name>`.
   This is the only step that proves the platform round trip — image
   resolution, sandbox create, exec, verifier, teardown — and nothing below it
   is claimable without one.
3. **A GPU rollout** if the provider is to appear in the
   [environments table](/user-guide/environments):
   `HARBOR_ENV_TYPE=<name>` with `tests/e2e/agentic/test_harbor_rollout.py`.
4. **A row in the table below**, which is where a claim about a
   (connector, provider) pair lives.

Nothing in step 1 needs a test change: the golden episode runs as one case per
registry entry, and the rollout e2e reads the provider from the environment.

## The two tests

| | `tests/e2e/agentic/test_sandbox_golden.py` | `tests/e2e/agentic/test_harbor_rollout.py` |
| --- | --- | --- |
| Proves | the platform round trip | harness, session server, TITO recording, reward |
| Needs | a provider credential | that, plus 2 GPUs and the model |
| Provider axis | one case per registry entry | `HARBOR_ENV_TYPE` |
| In CI | registered, skipped without a credential | registered, `disabled` |

The golden test drives `scripts/sandbox_smoke/run.py`, which stays the entry
point for the combinations CI does not fix — another agent harness, another
task, another connector. Its README covers those axes.

Neither runs in CI today: runners hold no sandbox credential. The golden case
starts running the day one lands, with no code change; the rollout e2e also
needs a route from the runner to the provider.

## What has been proven

| Connector | Provider | Golden episode | GPU rollout |
| --- | --- | --- | --- |
| Harbor | [E2B](https://e2b.dev/) (cloud or self-hosted) | 2026-09-02 | 2026-09-09 |
| Harbor | [Daytona](https://www.daytona.io/) | 2026-09-02 | — |
| Harbor | [Modal](https://modal.com/) | 2026-09-10 | — |

A dash is "not run", not "does not work": the registry reaches every provider
mechanically, and a pair earns a date by being run. OpenEnv's own golden path
is not wired into this driver yet; it lives in
`examples/experimental/openenv/scan_golden.py`.
