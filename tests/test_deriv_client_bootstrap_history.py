import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from configs.schema import DataIntegrityConfig, DerivConnectionConfig, HistoricalDataConfig
from data.deriv_client import DerivWebSocketClient
from data.integrity import IntegrityValidator


class _FakeBootstrapWebSocket:
    """
    Simulates a real server round-trip for ticks_history requests: `send()`
    parses the request and immediately enqueues the matching canned
    response, which the async iterator (consumed by `_listen()`) then
    yields on its own turn of the event loop — exercising the actual
    request -> _listen() -> _resolve_pending() -> future-resolves path,
    not a shortcut around it.
    """

    def __init__(self, candles_by_symbol: dict[str, list[dict]]):
        self._candles_by_symbol = candles_by_symbol
        self.sent_messages: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send(self, raw: str) -> None:
        request = json.loads(raw)
        self.sent_messages.append(request)
        symbol = request["ticks_history"]
        response = {
            "msg_type": "history",
            "req_id": request["req_id"],
            "candles": self._candles_by_symbol.get(symbol, []),
        }
        await self._queue.put(json.dumps(response))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        while True:
            msg = await self._queue.get()
            yield msg


def make_client():
    connection_config = DerivConnectionConfig(app_id="12345")
    historical_config = HistoricalDataConfig()
    validator = IntegrityValidator(DataIntegrityConfig())

    async def on_tick(tick):
        pass

    return DerivWebSocketClient(
        connection_config=connection_config,
        historical_config=historical_config,
        integrity_validator=validator,
        on_tick=on_tick,
    )


@pytest.mark.asyncio
async def test_fetch_bootstrap_history_returns_candles_per_symbol():
    fake_ws = _FakeBootstrapWebSocket(
        {
            "stpRNG": [{"epoch": 1000, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05}],
            "stpRNG2": [{"epoch": 1000, "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.05}],
        }
    )

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client()
    with patch("data.deriv_client.websockets.connect", fake_connect):
        results = await client.fetch_bootstrap_history(["stpRNG", "stpRNG2"], granularity_seconds=60)

    assert set(results.keys()) == {"stpRNG", "stpRNG2"}
    assert len(results["stpRNG"]) == 1
    assert results["stpRNG"][0].close == 1.05
    assert len(results["stpRNG2"]) == 1
    assert results["stpRNG2"][0].close == 2.05


@pytest.mark.asyncio
async def test_fetch_bootstrap_history_sends_one_request_per_symbol_with_granularity():
    fake_ws = _FakeBootstrapWebSocket({"stpRNG": [], "stpRNG2": []})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client()
    with patch("data.deriv_client.websockets.connect", fake_connect):
        await client.fetch_bootstrap_history(["stpRNG", "stpRNG2"], granularity_seconds=120)

    assert len(fake_ws.sent_messages) == 2
    symbols_requested = {m["ticks_history"] for m in fake_ws.sent_messages}
    assert symbols_requested == {"stpRNG", "stpRNG2"}
    assert all(m["granularity"] == 120 for m in fake_ws.sent_messages)


@pytest.mark.asyncio
async def test_fetch_bootstrap_history_clears_ws_reference_after_close():
    fake_ws = _FakeBootstrapWebSocket({"stpRNG": []})

    @asynccontextmanager
    async def fake_connect(url, **kwargs):
        yield fake_ws

    client = make_client()
    with patch("data.deriv_client.websockets.connect", fake_connect):
        await client.fetch_bootstrap_history(["stpRNG"])

    assert client._ws is None  # bootstrap connection fully torn down, run_forever() starts fresh
