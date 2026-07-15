"""
Live paper-trading pipeline orchestrator — the live equivalent of what
`WalkForwardBacktester` already does against historical data.

Per-candle flow (see `on_candle`):
  1. Settle any pending trade for this symbol first. Two genuinely
     different paths, chosen by whether the pending trade has a real
     `contract_id` (see `paper_trading.types.PendingTrade`):
       - Paper trade (`contract_id is None`): settled fictionally, using
         this candle's close vs the entry candle's close (see
         paper_trading_schema module docstring for why settlement is
         defined this way).
       - Live trade (`contract_id` set): Deriv, not this code, decides
         win/loss/payout on its own real `duration_ticks` clock. This
         polls `ContractOutcomeTracker` for that contract's real
         settlement; if it hasn't settled yet, the trade simply stays
         pending and NO new trade is opened on that symbol this candle
         (one open position per symbol at a time) — it is not, and must
         not be, resolved by the next-candle fiction.
  2. If nothing is still pending after step 1: classify regime ->
     estimate probability -> compute EV -> assess risk -> score the
     opportunity -> execute -> if bought, open a new pending trade
     (paper trades settle next candle; live trades await real
     settlement via step 1 on subsequent candles).

RL Trade Management (Level 7) is NOT wired in here — v1 holds every
paper trade to its one-candle settlement. See paper_trading_schema's
module docstring for the explicitly agreed v2 scope.

One `PaperTradingOrchestrator` instance owns ALL configured symbols
(rather than one per symbol), mirroring the account-level reality that
there is one risk/equity curve across every instrument traded, even
though each symbol gets its own probability model and opportunity
scorer (signal quality genuinely differs per instrument, unlike
capital, which is shared).
"""

from __future__ import annotations

import logging

import numpy as np

from configs.ev_schema import ExpectedValueConfig
from configs.execution_schema import ExecutionConfig
from configs.opportunity_schema import OpportunityScoringConfig
from configs.paper_trading_schema import PaperTradingConfig
from configs.post_trade_schema import PostTradeAnalysisConfig
from configs.probability_schema import BayesianLogisticConfig
from configs.risk_schema import RiskConfig
from data.types import Candle
from execution.engine import ExecutionEngine
from execution.outcome_tracker import ContractOutcomeTracker
from execution.types import BrokerClient
from expected_value.engine import ExpectedValueEngine
from expected_value.types import ContractSpec, ContractType
from features.types import FeatureVector
from opportunity.scorer import TradeOpportunityScorer
from paper_trading.types import PendingTrade, opportunity_to_pending_trade
from post_trade.analyzer import PostTradeAnalyzer
from post_trade.types import CompletedTrade, PerformanceMetrics
from probability.bayesian_logistic import BayesianLogisticRegression
from regime.rule_based import RuleBasedRegimeDetector
from risk.engine import RiskEngine
from risk.types import TradeOutcome
from state_encoder.encoder import MarketStateEncoder
from state_encoder.types import MarketState

logger = logging.getLogger("paper_trading")


class PaperTradingOrchestrator:
    def __init__(
        self,
        paper_config: PaperTradingConfig,
        probability_config: BayesianLogisticConfig,
        ev_config: ExpectedValueConfig,
        risk_config: RiskConfig,
        opportunity_config: OpportunityScoringConfig,
        post_trade_config: PostTradeAnalysisConfig,
        execution_config: ExecutionConfig,
        platform_environment: str,
        regime_detector: RuleBasedRegimeDetector,
        state_encoder: MarketStateEncoder,
        broker_client: BrokerClient | None = None,
    ) -> None:
        self._paper_config = paper_config
        self._probability_config = probability_config
        self._opportunity_config = opportunity_config
        self._regime_detector = regime_detector
        self._state_encoder = state_encoder

        self._ev_engine = ExpectedValueEngine(ev_config)
        self._risk_engine = RiskEngine(risk_config, starting_equity=paper_config.starting_equity)
        self._post_trade_analyzer = PostTradeAnalyzer(post_trade_config)
        self._execution_engine = ExecutionEngine(
            config=execution_config,
            platform_environment=platform_environment,
            broker_client=broker_client,
        )
        # Only needed once trades can actually go live (real contract_id
        # present) — no broker_client means every trade stays paper mode,
        # so nothing ever needs to poll for a real settlement.
        self._outcome_tracker = ContractOutcomeTracker(broker_client) if broker_client is not None else None
        self._contract = ContractSpec(
            contract_type=ContractType.RISE_FALL,
            stake=paper_config.stake,
            payout=paper_config.stake * paper_config.assumed_payout_ratio,
            duration_ticks=paper_config.duration_ticks,
        )

        self._probability_models: dict[str, BayesianLogisticRegression] = {}
        self._scorers: dict[str, TradeOpportunityScorer] = {}
        self._pending_trades: dict[str, PendingTrade | None] = {}

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #

    def bootstrap(self, symbol: str, historical_states: list[MarketState], closes: list[float]) -> bool:
        """
        Fits `symbol`'s initial probability model from historical
        (state, close) pairs — the caller is responsible for producing
        `historical_states` by replaying historical candles through the
        SAME live feature-pipeline/state-encoder instances that will keep
        running afterward, so rolling-window warm-up carries continuously
        into live streaming rather than resetting.

        Returns True if enough usable states were available to fit (per
        `paper_config.min_bootstrap_candles`); False (logged) if not — the
        symbol is simply skipped until enough live candles accumulate to
        retry via a later `bootstrap()` call.
        """
        if len(historical_states) != len(closes):
            raise ValueError("historical_states and closes must be the same length")

        if len(historical_states) < self._paper_config.min_bootstrap_candles + 1:
            logger.warning(
                "%s: only %d usable historical states available (need >= %d) — "
                "skipping bootstrap for now.",
                symbol, len(historical_states), self._paper_config.min_bootstrap_candles + 1,
            )
            return False

        usable_states = historical_states[:-1]
        labels = (np.diff(np.array(closes)) > 0).astype(int)
        X = np.array(
            [[getattr(s, dim) for dim in self._probability_config.feature_dims] for s in usable_states]
        )

        model = BayesianLogisticRegression(self._probability_config)
        model.fit(X, labels)

        self._probability_models[symbol] = model
        self._scorers[symbol] = TradeOpportunityScorer(self._opportunity_config)
        self._pending_trades[symbol] = None
        logger.info("%s: bootstrap fit complete on %d historical states.", symbol, len(usable_states))
        return True

    def is_bootstrapped(self, symbol: str) -> bool:
        return symbol in self._probability_models

    # ------------------------------------------------------------------ #
    # Live per-candle decision loop
    # ------------------------------------------------------------------ #

    async def on_candle(self, symbol: str, candle: Candle, vector: FeatureVector | None) -> dict:
        """
        Called once per completed candle for `symbol`. `vector` is whatever
        the live feature pipeline produced for this candle (None if still
        warming up on rolling windows — nothing to do yet either way).

        Returns {"settled": CompletedTrade | None, "decision": ExecutionDecision | None}
        for logging; never raises for ordinary "nothing to do" cases.
        """
        result: dict = {"settled": None, "decision": None}

        if symbol not in self._probability_models:
            return result

        pending = self._pending_trades.get(symbol)
        if pending is not None and pending.is_awaiting_real_settlement:
            settled_trade = await self._poll_live_settlement_if_any(symbol, pending, candle)
        else:
            settled_trade = self._settle_pending_trade_if_any(symbol, candle)
        result["settled"] = settled_trade

        if self._pending_trades.get(symbol) is not None:
            # A live trade is still awaiting its real Deriv settlement —
            # never open a second concurrent position on the same symbol
            # while one is outstanding.
            return result

        if vector is None:
            return result

        state = self._state_encoder.encode(vector)
        if not state.is_valid:
            return result

        regime = self._regime_detector.classify(state)
        probability = self._probability_models[symbol].predict(state)
        ev = self._ev_engine.evaluate(probability, self._contract)
        risk = self._risk_engine.assess(ev)
        opportunity = self._scorers[symbol].evaluate(ev, risk, regime, probability)
        decision = await self._execution_engine.execute(opportunity, ev, risk, self._contract)
        result["decision"] = decision

        if decision.action == "buy":
            self._pending_trades[symbol] = opportunity_to_pending_trade(
                opportunity,
                entry_close=candle.close,
                direction=ev.direction,
                stake=decision.stake,
                payout=decision.payout,
                probability_used=ev.probability_used,
                contract_id=decision.contract_id,
            )

        return result

    def _settle_pending_trade_if_any(self, symbol: str, candle: Candle) -> CompletedTrade | None:
        """Paper-mode settlement: fictional next-candle-close direction.
        Never called for a trade with a real contract_id — see on_candle."""
        pending = self._pending_trades.get(symbol)
        if pending is None:
            return None

        actual_direction = 1 if candle.close > pending.entry_close else -1
        won = actual_direction == pending.direction
        pnl = (pending.payout - pending.stake) if won else -pending.stake

        return self._finalize_settled_trade(
            symbol, pending, exit_epoch=candle.epoch, pnl=pnl, exit_reason="settled_next_candle_close",
        )

    async def _poll_live_settlement_if_any(
        self, symbol: str, pending: PendingTrade, candle: Candle
    ) -> CompletedTrade | None:
        """
        Live-mode settlement: ask Deriv, via `ContractOutcomeTracker`, what
        actually happened to `pending.contract_id` — never fabricate an
        outcome the way paper mode does. If the contract hasn't settled
        yet (or the poll itself fails), the trade simply stays pending and
        this returns None; `on_candle` will try again on the next candle.
        """
        assert self._outcome_tracker is not None  # guaranteed whenever a pending trade has a contract_id

        try:
            outcome = await self._outcome_tracker.poll(pending.contract_id)
        except Exception as exc:  # noqa: BLE001 — broker/network failures are logged, not raised
            logger.warning(
                "%s: contract status poll failed for contract_id=%s: %s — will retry next candle.",
                symbol, pending.contract_id, exc,
            )
            return None

        if not outcome.is_sold:
            return None

        return self._finalize_settled_trade(
            symbol, pending, exit_epoch=candle.epoch, pnl=outcome.pnl, exit_reason="broker_settled",
        )

    def _finalize_settled_trade(
        self, symbol: str, pending: PendingTrade, exit_epoch: int, pnl: float, exit_reason: str,
    ) -> CompletedTrade:
        """
        Shared by both settlement paths: feed the real pnl (whichever
        source it came from) into the Risk Engine's equity/circuit-breaker
        tracking and the Post-Trade Analyzer, then clear the pending slot
        for this symbol so a new trade can be opened.
        """
        new_equity = self._risk_engine.equity + pnl
        self._risk_engine.record_trade_result(
            TradeOutcome(epoch=exit_epoch, pnl=pnl, equity_after=new_equity)
        )
        trade = CompletedTrade(
            symbol=symbol,
            entry_epoch=pending.entry_epoch,
            exit_epoch=exit_epoch,
            direction=pending.direction,
            stake=pending.stake,
            pnl=pnl,
            predicted_probability=pending.predicted_probability,
            regime_at_entry=pending.regime_at_entry,
            quality_score_at_entry=pending.quality_score_at_entry,
            exit_reason=exit_reason,
        )
        self._post_trade_analyzer.record_trade(trade)
        self._pending_trades[symbol] = None
        return trade

    # ------------------------------------------------------------------ #
    # Monitoring accessors
    # ------------------------------------------------------------------ #

    @property
    def equity(self) -> float:
        return self._risk_engine.equity

    def metrics(self, window: int | None = None) -> PerformanceMetrics:
        return self._post_trade_analyzer.compute_metrics(window=window)

    @property
    def n_trades_recorded(self) -> int:
        return self._post_trade_analyzer.n_trades_recorded
