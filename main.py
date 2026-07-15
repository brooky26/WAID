"""
Entry point for the platform.

Usage:
    python main.py --config configs/default.yaml

Pipeline: Deriv ticks -> integrity validation -> storage (raw ticks) ->
candle aggregation -> feature engineering -> state encoding -> regime
classification -> [if paper_trading.enabled] Probability -> EV -> Risk ->
Opportunity Scoring -> Execution (paper mode) -> settlement -> Post-Trade.

Before live streaming starts, if paper_trading.enabled, each configured
symbol's probability model is bootstrap-fit from historical candles
(fetched via a short-lived connection — see
DerivWebSocketClient.fetch_bootstrap_history) replayed through the SAME
feature-pipeline/state-encoder instances that continue into live
streaming afterward, so rolling-window warm-up carries over continuously
rather than resetting. Symbols with too little historical data are
skipped (logged) rather than silently left broken.

Scope/simplifications of this pipeline (v1) are documented in
configs/paper_trading_schema.py's module docstring — most importantly:
settlement is next-candle-close direction (not real duration_ticks
expiry), payout is an assumed ratio (not a live Deriv proposal), and
Level 7 (RL Trade Management) is not wired in — every trade is held to
its one-candle settlement. Set paper_trading.enabled: false in config to
fall back to the original data-collection + regime-logging-only mode.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from configs.loader import load_config
from data.candle_aggregator import CandleAggregator
from data.deriv_client import DerivWebSocketClient
from data.integrity import IntegrityValidator
from data.storage import SQLiteTickStore, SupabaseTickStore, TickStore
from data.types import Candle, ConnectionEvent, Tick
from features.pipeline import FeatureEngineeringPipeline
from features.types import FeatureVector
from paper_trading.orchestrator import PaperTradingOrchestrator
from regime.rule_based import RuleBasedRegimeDetector
from regime.types import RegimeClassification
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


def build_tick_store(storage_cfg) -> TickStore:
    """Backend is chosen by configs.market_data.storage.backend (sqlite or
    supabase — see StorageConfig). sqlite is the local-dev default; supabase
    is required for Railway, since a local sqlite file does not survive a
    redeploy/restart on Railway's ephemeral filesystem."""
    if storage_cfg.backend == "sqlite":
        return SQLiteTickStore(
            db_path=storage_cfg.sqlite_path,
            write_batch_size=storage_cfg.write_batch_size,
            flush_interval_seconds=storage_cfg.flush_interval_seconds,
        )
    elif storage_cfg.backend == "supabase":
        return SupabaseTickStore(
            supabase_url=storage_cfg.supabase_url,
            supabase_key=storage_cfg.supabase_key,
            table=storage_cfg.supabase_table,
            write_batch_size=storage_cfg.write_batch_size,
            flush_interval_seconds=storage_cfg.flush_interval_seconds,
        )
    raise ValueError(f"Unknown storage backend: {storage_cfg.backend}")


async def bootstrap_paper_trading(
    client: DerivWebSocketClient,
    orchestrator: PaperTradingOrchestrator,
    feature_pipeline: FeatureEngineeringPipeline,
    state_encoder: MarketStateEncoder,
    symbols: list[str],
    granularity_seconds: int,
) -> None:
    logger.info("Bootstrapping probability models from historical candles for: %s", symbols)
    historical = await client.fetch_bootstrap_history(symbols, granularity_seconds)

    for symbol, candles in historical.items():
        states: list[MarketState] = []
        closes: list[float] = []
        for candle in candles:
            vector = feature_pipeline.on_candle(candle)
            if vector is not None:
                state = state_encoder.encode(vector)
                if state.is_valid:
                    states.append(state)
                    closes.append(candle.close)

        fitted = orchestrator.bootstrap(symbol, states, closes)
        if not fitted:
            logger.warning(
                "%s: skipped for live trading (insufficient bootstrap history); "
                "data collection continues, will retry once enough live candles accumulate.",
                symbol,
            )


async def run(config_path: str) -> None:
    config = load_config(config_path)
    md_cfg = config.market_data
    feature_cfg = config.feature_engineering
    paper_cfg = config.paper_trading

    store = build_tick_store(md_cfg.storage)
    await store.start()

    validator = IntegrityValidator(md_cfg.integrity)
    aggregator = CandleAggregator(
        granularity_seconds=md_cfg.historical.candle_granularity_seconds
    )
    feature_pipeline = FeatureEngineeringPipeline(feature_cfg)
    state_encoder = MarketStateEncoder(config.state_encoder)
    # Rule-based detector is the active default: it needs no training data
    # and is available from the first valid MarketState. The Gaussian HMM
    # (regime/hmm_detector.py) is a challenger — fit it offline on
    # accumulated historical MarketState vectors, validate it, and only
    # then consider swapping it in via the (future) Champion-Challenger
    # framework. Wiring an unfit HMM in here would just raise on first use.
    regime_detector = RuleBasedRegimeDetector(config.regime_detection.rule_based)

    async def on_connection_event(event: ConnectionEvent) -> None:
        logger.info("Connection event: %s (%s)", event.event, event.detail)

    orchestrator: PaperTradingOrchestrator | None = None

    def on_regime(classification: RegimeClassification) -> None:
        status = "valid" if classification.is_valid else "invalid (NaN state)"
        logger.info(
            "Regime [%s @ %d] %s — %s (confidence=%.2f, detector=%s)",
            classification.symbol,
            classification.epoch,
            status,
            classification.regime.value,
            classification.confidence,
            classification.detector_name,
        )

    async def on_candle(candle: Candle) -> None:
        vector = feature_pipeline.on_candle(candle)
        if vector is not None:
            state = state_encoder.encode(vector)
            on_regime(regime_detector.classify(state))

        if orchestrator is None:
            return

        result = await orchestrator.on_candle(candle.symbol, candle, vector)
        settled = result["settled"]
        if settled is not None:
            logger.info(
                "SETTLED %s: %s pnl=%.2f (equity=%.2f, total_trades=%d)",
                settled.symbol,
                "WIN" if settled.was_win else "LOSS",
                settled.pnl,
                orchestrator.equity,
                orchestrator.n_trades_recorded,
            )
        decision = result["decision"]
        if decision is not None and decision.action != "skip":
            logger.info("DECISION %s: %s (%s)", decision.symbol, decision.action, decision.reason)

    async def on_tick(tick: Tick) -> None:
        await store.write_ticks([tick])
        completed_candle = aggregator.on_tick(tick)
        if completed_candle is not None:
            await on_candle(completed_candle)

    client = DerivWebSocketClient(
        connection_config=md_cfg.connection,
        historical_config=md_cfg.historical,
        integrity_validator=validator,
        on_tick=on_tick,
        on_connection_event=on_connection_event,
    )

    if paper_cfg.enabled:
        orchestrator = PaperTradingOrchestrator(
            paper_config=paper_cfg,
            probability_config=config.probability_estimation.bayesian_logistic,
            ev_config=config.expected_value,
            risk_config=config.risk,
            opportunity_config=config.opportunity_scoring,
            post_trade_config=config.post_trade_analysis,
            execution_config=config.execution,
            platform_environment=config.environment,
            regime_detector=regime_detector,
            state_encoder=state_encoder,
            broker_client=client,
        )

    try:
        if orchestrator is not None:
            await bootstrap_paper_trading(
                client, orchestrator, feature_pipeline, state_encoder,
                md_cfg.connection.symbols, md_cfg.historical.candle_granularity_seconds,
            )
        await client.run_forever()
    finally:
        await client.stop()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv Trading Research Platform")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
