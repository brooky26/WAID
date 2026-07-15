"""
Deriv WebSocket Client — Market Data Layer entry point.

Connection layer: REST OTP bootstrap (current Deriv Options API)
-------------------------------------------------------------------
Deriv retired the legacy pattern of connecting directly to
`wss://ws.derivws.com/websockets/v3?app_id=...`. The current flow is:

  Public / unauthenticated (market data only):
      connect directly to `ws_public_url`
      (wss://api.derivws.com/trading/v1/options/ws/public)

  Authenticated (needed for anything beyond public ticks/candles):
      1. POST {rest_base_url}/trading/v1/options/accounts/{account_id}/otp
         headers: Deriv-App-ID: <app_id>, Authorization: Bearer <api_token>
      2. Response: {"data": {"url": "wss://.../ws/demo?otp=..."}}
      3. Connect directly to that URL.

OTP tokens are short-lived. This client requests a fresh OTP on every
reconnect (not just the first connect) — reusing a stale OTP after a
drop will fail authentication.

Everything downstream of "how did we get a URL to connect to" —
subscription messages, tick parsing, reconnect backoff — is unchanged
from before; the message-level JSON-RPC schema (ticks/ticks_history
requests, tick/history responses) is still what Deriv's WS speaks, just
over the new transport, with tick.symbol/epoch/quote now guaranteed
present in every tick message (previously optional).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from configs.schema import DerivConnectionConfig, HistoricalDataConfig
from data.integrity import IntegrityValidator
from data.types import Candle, ConnectionEvent, DataQualityFlag, Tick

logger = logging.getLogger(__name__)

TickCallback = Callable[[Tick], Awaitable[None]]
ConnectionEventCallback = Callable[[ConnectionEvent], Awaitable[None]]


class DerivClientError(Exception):
    """Raised for unrecoverable protocol-level errors (e.g. bad app_id, OTP failure)."""


class DerivOTPBootstrap:
    """
    Handles the REST leg of the connection: exchanging an api_token for a
    fresh, ready-to-use authenticated WebSocket URL. Isolated from the
    WebSocket client itself so it's independently testable (mock the HTTP
    call, no live socket needed) and reusable by other modules later
    (e.g. execution) that also need an authenticated connection.
    """

    def __init__(self, config: DerivConnectionConfig) -> None:
        self._config = config

    async def fetch_authenticated_ws_url(self) -> str:
        if not self._config.is_authenticated_mode:
            raise DerivClientError(
                "fetch_authenticated_ws_url called without api_token/account_id configured."
            )
        endpoint = (
            f"{self._config.rest_base_url}/trading/v1/options/accounts/"
            f"{self._config.account_id}/otp"
        )
        headers = {
            "Deriv-App-ID": self._config.app_id,
            "Authorization": f"Bearer {self._config.api_token}",
        }
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers) as resp:
                payload = await resp.json()
                if resp.status != 200:
                    errors = payload.get("errors", payload)
                    raise DerivClientError(
                        f"OTP request failed (status={resp.status}): {errors}"
                    )
                url = payload.get("data", {}).get("url")
                if not url:
                    raise DerivClientError(
                        f"OTP response missing data.url: {payload}"
                    )
                return url


class DerivWebSocketClient:
    def __init__(
        self,
        connection_config: DerivConnectionConfig,
        historical_config: HistoricalDataConfig,
        integrity_validator: IntegrityValidator,
        on_tick: TickCallback,
        on_connection_event: ConnectionEventCallback | None = None,
        otp_bootstrap: DerivOTPBootstrap | None = None,
    ) -> None:
        self._cfg = connection_config
        self._hist_cfg = historical_config
        self._validator = integrity_validator
        self._on_tick = on_tick
        self._on_connection_event = on_connection_event
        self._otp = otp_bootstrap or DerivOTPBootstrap(connection_config)
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id_counter = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._running = False
        self._reconnect_attempt = 0

    # ------------------------------------------------------------------ #
    # Public lifecycle
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        """Connect, subscribe, stream ticks; reconnect automatically on drop."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                await self._emit_event("disconnected", detail=str(exc))
                if not self._running:
                    break
                should_continue = await self._handle_reconnect()
                if not should_continue:
                    break
            except DerivClientError:
                raise

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()

    # ------------------------------------------------------------------ #
    # Connection + subscription
    # ------------------------------------------------------------------ #

    async def _resolve_connect_url(self) -> str:
        """
        Public mode: connect straight to ws_public_url.
        Authenticated mode: fetch a *fresh* OTP-embedded URL every time
        this is called — never reuse a URL from a previous connection,
        since OTP tokens are short-lived and reconnects need a new one.
        """
        if not self._cfg.is_authenticated_mode:
            return self._cfg.ws_public_url
        return await self._otp.fetch_authenticated_ws_url()

    async def _connect_and_stream(self) -> None:
        url = await self._resolve_connect_url()
        async with websockets.connect(
            url, ping_interval=self._cfg.ping_interval_seconds
        ) as ws:
            self._ws = ws
            self._reconnect_attempt = 0
            await self._emit_event("connected", detail=self._redact(url))

            for symbol in self._cfg.symbols:
                await self._subscribe_ticks(ws, symbol)

            await self._listen(ws)

    @staticmethod
    def _redact(url: str) -> str:
        """Never log a live OTP token."""
        if "otp=" in url:
            base, _, _ = url.partition("otp=")
            return f"{base}otp=<redacted>"
        return url

    async def _subscribe_ticks(self, ws, symbol: str) -> None:
        req_id = self._next_req_id()
        await ws.send(
            json.dumps({"ticks": symbol, "subscribe": 1, "req_id": req_id})
        )

    async def _listen(self, ws) -> None:
        async for raw_message in ws:
            message = json.loads(raw_message)

            if message.get("error"):
                logger.warning(
                    "Deriv API error (msg_type=%s, echo_req=%s): %s",
                    message.get("msg_type"),
                    message.get("echo_req"),
                    message["error"],
                )
                self._resolve_pending(message)  # also resolve pending requests with the error
                continue

            msg_type = message.get("msg_type")
            if msg_type == "tick":
                await self._handle_tick_message(message)
            elif msg_type in ("history", "candles", "proposal", "buy", "proposal_open_contract"):
                self._resolve_pending(message)

    async def _handle_tick_message(self, message: dict) -> None:
        # Error responses to a tick subscription still arrive with
        # msg_type == "tick" but no "tick" payload — those are caught by the
        # error check in _listen before this is called, but guard here too
        # in case a symbol is delisted/invalid mid-stream and Deriv sends a
        # tick-shaped message without the field regardless.
        tick_data = message.get("tick")
        if not tick_data:
            logger.warning("Received tick-typed message with no tick payload: %s", message)
            return
        raw_tick = Tick(
            symbol=tick_data["symbol"],
            epoch=int(tick_data["epoch"]),
            quote=float(tick_data["quote"]),
            received_at=datetime.now(timezone.utc),
        )
        validated = self._validator.validate(raw_tick)
        if validated.quality == DataQualityFlag.GAP_DETECTED:
            logger.info(
                "Gap detected for %s at epoch %d — backfill should be triggered.",
                validated.symbol,
                validated.epoch,
            )
        await self._on_tick(validated)

    # ------------------------------------------------------------------ #
    # Historical backfill
    # ------------------------------------------------------------------ #

    async def fetch_bootstrap_history(
        self, symbols: list[str], granularity_seconds: int | None = None
    ) -> dict[str, list[Candle]]:
        """
        Fetch historical candles for multiple symbols via a short-lived
        connection, independent of `run_forever()`'s long-lived streaming
        connection. Exists specifically for pipeline bootstrap: fitting a
        symbol's initial probability model needs historical candles *before*
        live tick streaming starts, but `run_forever()` connects, subscribes
        ticks, and blocks in `_listen()` — it has no hook to "just fetch
        history and return." This opens its own connection, fetches
        everything requested, and closes it; `run_forever()` opens its own
        fresh connection afterward as normal, so this has zero effect on the
        streaming connection's lifecycle, reconnect behavior, or OTP token
        usage (OTP tokens are single-use anyway, so a fresh one here is
        correct, not wasteful).

        `fetch_historical_candles` only resolves once `_listen()` reads the
        matching "history" response off the socket and resolves the pending
        future — so a background `_listen()` task has to be running
        concurrently with the requests below, not just the bare connection.
        """
        url = await self._resolve_connect_url()
        results: dict[str, list[Candle]] = {}
        async with websockets.connect(url, ping_interval=None) as ws:
            self._ws = ws
            listen_task = asyncio.create_task(self._listen(ws))
            try:
                for symbol in symbols:
                    results[symbol] = await self.fetch_historical_candles(symbol, granularity_seconds)
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
        self._ws = None
        return results

    async def fetch_historical_candles(
        self, symbol: str, granularity_seconds: int | None = None
    ) -> list[Candle]:
        """One-shot request/response call, independent of the streaming loop."""
        if self._ws is None:
            raise DerivClientError("Cannot fetch history: not connected.")

        granularity = granularity_seconds or self._hist_cfg.candle_granularity_seconds
        req_id = self._next_req_id()
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": self._hist_cfg.request_count_max,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": granularity,
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        response = await asyncio.wait_for(
            future, timeout=self._cfg.request_timeout_seconds
        )
        if response.get("error"):
            raise DerivClientError(f"History request failed: {response['error']}")

        candles_raw = response.get("candles", [])
        return [
            Candle(
                symbol=symbol,
                epoch=int(c["epoch"]),
                granularity=granularity,
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
            )
            for c in candles_raw
        ]

    def _resolve_pending(self, message: dict) -> None:
        req_id = message.get("req_id")
        if req_id in self._pending:
            self._pending.pop(req_id).set_result(message)

    # ------------------------------------------------------------------ #
    # Execution: live proposal + buy (Level 6)
    # ------------------------------------------------------------------ #

    async def fetch_proposal(
        self,
        symbol: str,
        contract_type_code: str,
        stake: float,
        duration_ticks: int,
        currency: str = "USD",
    ) -> dict:
        """
        Request a live price quote for a contract. `contract_type_code` is
        Deriv's own code (e.g. "CALL"/"PUT" for Rise/Fall) — mapping from
        our internal ContractType/direction to Deriv's codes is the
        execution engine's job (execution/engine.py), not this client's;
        this method is a thin, honest wrapper over the wire protocol only.

        Returns the raw `proposal` dict from Deriv's response (id,
        ask_price, payout, ...) — the caller decides what to do with it.
        """
        if self._ws is None:
            raise DerivClientError("Cannot fetch proposal: not connected.")

        req_id = self._next_req_id()
        request = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type_code,
            "currency": currency,
            "duration": duration_ticks,
            "duration_unit": "t",
            "symbol": symbol,
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        response = await asyncio.wait_for(
            future, timeout=self._cfg.request_timeout_seconds
        )
        if response.get("error"):
            raise DerivClientError(f"Proposal request failed: {response['error']}")
        return response["proposal"]

    async def buy(self, proposal_id: str, price: float) -> dict:
        """
        Execute a buy against a previously-fetched proposal. Returns the
        raw `buy` dict from Deriv's response (contract_id, buy_price,
        payout, ...).

        This is the one method in the entire codebase that spends real
        money when connected to a real (non-demo) account — it does
        exactly what it's told and nothing more; every safety decision
        (should we buy at all, at what stake, is this proposal stale)
        belongs upstream in execution/engine.py, not here.
        """
        if self._ws is None:
            raise DerivClientError("Cannot buy: not connected.")

        req_id = self._next_req_id()
        request = {"buy": proposal_id, "price": price, "req_id": req_id}
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        response = await asyncio.wait_for(
            future, timeout=self._cfg.request_timeout_seconds
        )
        if response.get("error"):
            raise DerivClientError(f"Buy request failed: {response['error']}")
        return response["buy"]

    async def check_contract_status(self, contract_id: str) -> dict:
        """
        One-shot poll of a previously-bought contract's real status via
        Deriv's `proposal_open_contract` call. Deliberately `subscribe: 0`
        — a real push subscription would need its own long-lived
        per-contract stream management (and cleanup on settlement, and
        reconnect handling), which is more machinery than a poll-on-some-
        cadence caller (the orchestrator, once per candle) actually needs.
        If push-based settlement latency ever matters enough to justify
        that complexity, this is the method to extend, not replace —
        callers only depend on "give me the current status right now".

        Returns the raw `proposal_open_contract` dict from Deriv's
        response (is_sold, profit, payout, buy_price, sell_price,
        status, ...) unmodified — interpreting what "settled" means is
        the caller's job (see execution/outcome_tracker.py), not this
        client's.
        """
        if self._ws is None:
            raise DerivClientError("Cannot check contract status: not connected.")

        req_id = self._next_req_id()
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 0,
            "req_id": req_id,
        }
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self._ws.send(json.dumps(request))

        response = await asyncio.wait_for(
            future, timeout=self._cfg.request_timeout_seconds
        )
        if response.get("error"):
            raise DerivClientError(f"Contract status request failed: {response['error']}")
        return response["proposal_open_contract"]

    # ------------------------------------------------------------------ #
    # Reconnect logic
    # ------------------------------------------------------------------ #

    async def _handle_reconnect(self) -> bool:
        if (
            self._cfg.max_reconnect_attempts is not None
            and self._reconnect_attempt >= self._cfg.max_reconnect_attempts
        ):
            await self._emit_event(
                "reconnect_failed", detail="max_reconnect_attempts exhausted"
            )
            return False

        backoff = min(
            self._cfg.reconnect_initial_backoff_seconds
            * (self._cfg.reconnect_backoff_multiplier ** self._reconnect_attempt),
            self._cfg.reconnect_max_backoff_seconds,
        )
        self._reconnect_attempt += 1
        detail = f"attempt {self._reconnect_attempt}, waiting {backoff:.1f}s"
        if self._cfg.is_authenticated_mode:
            detail += " (will request a fresh OTP before reconnecting)"
        await self._emit_event("reconnecting", detail=detail)
        await asyncio.sleep(backoff)
        return True

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _next_req_id(self) -> int:
        self._req_id_counter += 1
        return self._req_id_counter

    async def _emit_event(self, event: str, detail: str) -> None:
        logger.info("Deriv client event: %s — %s", event, detail)
        if self._on_connection_event is not None:
            await self._on_connection_event(
                ConnectionEvent(
                    event=event,
                    attempt=self._reconnect_attempt,
                    detail=detail,
                    occurred_at=datetime.now(timezone.utc),
                )
            )
