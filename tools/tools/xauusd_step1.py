#!/usr/bin/env python3
"""Step 1: minimal XAUUSD backtest with Backtrader (without pandas)."""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from collections import deque
from pathlib import Path

import backtrader as bt


class Step1SimpleStrategy(bt.Strategy):
    """EMA-cross strategy with toggleable RSI/Volume filters and per-filter TF."""

    params = dict(
        enable_longs=True,
        enable_shorts=False,
        use_ema_filter=True,
        ema_period=50,
        ema_condition="below_after_above",
        ema_min_above_hours=2.0,
        use_rsi=False,
        rsi_len=50,
        rsi_min=20.0,
        rsi_max=80.0,
        use_volume=False,
        vol_len=50,
        vol_multiplier=1.5,
        use_short_ema_filter=True,
        short_ema_period=50,
        short_ema_condition="above_after_below",
        short_ema_min_below_hours=2.0,
        use_short_rsi=False,
        short_rsi_len=50,
        short_rsi_min=20.0,
        short_rsi_max=80.0,
        ema_data_idx=0,
        rsi_data_idx=0,
        vol_data_idx=0,
        short_ema_data_idx=0,
        short_rsi_data_idx=0,
        short_rsi_filters=(),
        short_rsi_filter_data_idxs=(),
        # Pine-like entry sizing and pyramiding
        entry_cash=500.0,
        entry_cash_buffer=0.98,
        max_entries=3,
        min_add_minutes=5,
        add_only_lower_price=True,
        min_add_drop_pct=0.0,
        rsi_add_drop_pct=0.0,
        # Pine-like trailing USDT logic
        trail_activation_usdt=4.0,
        initial_stoploss_usdt=-20.0,
        short_initial_stoploss_usdt=-20.0,
        trail_keep_pct=0.5,
        trail_activation_offset_usdt=0.0,
        use_tp_ladder=False,
        use_short_tp_ladder=False,
        tp_levels=0,
        tp_step_pct=10.0,
        tp_close_pct=10.0,
        verbose_trades=True,
    )

    def __init__(self) -> None:
        self.trade_data = self.datas[0]
        self.ema_data = self.datas[self.p.ema_data_idx]
        self.rsi_data = self.datas[self.p.rsi_data_idx]
        self.vol_data = self.datas[self.p.vol_data_idx]
        self.short_ema_data = self.datas[self.p.short_ema_data_idx]
        self.short_rsi_data = self.datas[self.p.short_rsi_data_idx]

        self.ema = bt.ind.EMA(self.ema_data.close, period=self.p.ema_period)
        self.short_ema = bt.ind.EMA(self.short_ema_data.close, period=self.p.short_ema_period)

        self.rsi = bt.ind.RSI_Safe(self.rsi_data.close, period=self.p.rsi_len)
        self.short_rsi = bt.ind.RSI_Safe(self.short_rsi_data.close, period=self.p.short_rsi_len)
        self.short_rsi_filters_cfg = list(self.p.short_rsi_filters or [])
        self.short_rsi_filter_data_idxs = list(self.p.short_rsi_filter_data_idxs or [])
        self.short_rsi_filter_inds = []
        for idx, cfg in enumerate(self.short_rsi_filters_cfg):
            data_idx = self.short_rsi_filter_data_idxs[idx]
            data = self.datas[data_idx]
            ind = bt.ind.RSI_Safe(data.close, period=int(cfg["len"]))
            ind.plotinfo.plot = False
            self.short_rsi_filter_inds.append((ind, float(cfg["min"]), float(cfg["max"])))
        self.rsi.plotlines.rsi._linewidth = 0.5
        self.vol_sma = bt.ind.SMA(self.vol_data.volume, period=self.p.vol_len)
        self.vol_sma.plotinfo.plot = False

        self.order = None
        self.trade_count = 0
        self.winning_trades = 0
        self.long_trade_count = 0
        self.short_trade_count = 0
        self.long_pnl = 0.0
        self.short_pnl = 0.0
        self.first_trade_open_dt = None
        self.last_trade_close_dt = None
        self.entry_signal_count = 0
        self.rejected_orders = 0
        self.entry_count = 0
        self.last_entry_dt = None
        self.last_entry_price = None
        self.last_entry_rsi = None
        self.last_short_entry_dt = None
        self.last_short_entry_price = None
        self.last_short_entry_rsi = None

        self.max_profit_usdt = 0.0
        self.stoploss_usdt = self.p.initial_stoploss_usdt
        self.trailing_active = False
        self.tp_hits = 0
        self.next_tp_target_usdt = None
        self.cycle_ref_size = 0.0

        self._last_ema_len = 0
        self._ema_above_streak = 0
        self._ema_drop_armed = False
        self._ema_signal_active = False
        self._last_short_ema_len = 0
        self._short_ema_below_streak = 0
        self._short_ema_rise_armed = False
        self._short_ema_signal_active = False

    def _current_dt(self, data: bt.feed.DataBase | None = None) -> str:
        d = data if data is not None else self.trade_data
        return d.datetime.datetime(0).strftime("%Y-%m-%d %H:%M:%S")

    def _entry_size(self) -> float:
        close = self.trade_data.close[0]
        if close <= 0:
            return 0.0
        cash_to_use = min(self.p.entry_cash, self.broker.getcash()) * self.p.entry_cash_buffer
        if cash_to_use <= 0:
            return 0.0
        return cash_to_use / close


    def _ema_ok(self) -> bool:
        if not self.p.use_ema_filter:
            return True

        if self.p.ema_condition == "above":
            return self.ema_data.close[0] > self.ema[0]
        if self.p.ema_condition == "below":
            return self.ema_data.close[0] < self.ema[0]

        # below_after_above: price must have been above EMA for at least X hours,
        # then moved below. Signal stays active while below.
        if self.p.ema_condition == "below_after_above":
            return self._ema_signal_active

        return self.ema_data.close[0] > self.ema[0]

    def _short_ema_ok(self) -> bool:
        if not self.p.use_short_ema_filter:
            return True

        if self.p.short_ema_condition == "below":
            return self.short_ema_data.close[0] < self.short_ema[0]
        if self.p.short_ema_condition == "above":
            return self.short_ema_data.close[0] > self.short_ema[0]
        if self.p.short_ema_condition == "above_after_below":
            return self._short_ema_signal_active
        return self.short_ema_data.close[0] < self.short_ema[0]


    def _update_ema_state(self) -> None:
        if len(self.ema_data) == self._last_ema_len:
            return

        self._last_ema_len = len(self.ema_data)

        required_bars = max(1, int(round((self.p.ema_min_above_hours * 60.0) / self.ema_data._compression)))
        is_above = self.ema_data.close[0] > self.ema[0]

        if is_above:
            self._ema_above_streak += 1
            if self._ema_above_streak >= required_bars:
                self._ema_drop_armed = True
            self._ema_signal_active = False
        else:
            if self._ema_drop_armed:
                self._ema_signal_active = True
            self._ema_above_streak = 0

    def _update_short_ema_state(self) -> None:
        if len(self.short_ema_data) == self._last_short_ema_len:
            return

        self._last_short_ema_len = len(self.short_ema_data)
        required_bars = max(1, int(round((self.p.short_ema_min_below_hours * 60.0) / self.short_ema_data._compression)))
        is_below = self.short_ema_data.close[0] < self.short_ema[0]

        if is_below:
            self._short_ema_below_streak += 1
            if self._short_ema_below_streak >= required_bars:
                self._short_ema_rise_armed = True
            self._short_ema_signal_active = False
        else:
            if self._short_ema_rise_armed:
                self._short_ema_signal_active = True
            self._short_ema_below_streak = 0

    def _rsi_ok(self) -> bool:
        if not self.p.use_rsi:
            return True
        rsi_val = self.rsi[0]
        return self.p.rsi_min < rsi_val < self.p.rsi_max

    def _volume_ok(self) -> bool:
        if not self.p.use_volume:
            return True
        vol_sma = self.vol_sma[0]
        if vol_sma <= 0:
            return False
        return self.vol_data.volume[0] > vol_sma * self.p.vol_multiplier

    def _short_rsi_ok(self) -> bool:
        if self.short_rsi_filter_inds:
            return all(rsi_min < ind[0] < rsi_max for ind, rsi_min, rsi_max in self.short_rsi_filter_inds)
        if not self.p.use_short_rsi:
            return True
        val = self.short_rsi[0]
        return self.p.short_rsi_min < val < self.p.short_rsi_max

    def _short_rsi_ref_value(self) -> float | None:
        if self.short_rsi_filter_inds:
            return float(self.short_rsi_filter_inds[0][0][0])
        if self.p.use_short_rsi:
            return float(self.short_rsi[0])
        return None


    def _can_add_entry(self) -> bool:
        if self.entry_count <= 0:
            return True

        if self.p.min_add_minutes > 0 and self.last_entry_dt is not None:
            now_dt = self.trade_data.datetime.datetime(0)
            elapsed_mins = (now_dt - self.last_entry_dt).total_seconds() / 60.0
            if elapsed_mins < self.p.min_add_minutes:
                return False

        if self.p.add_only_lower_price and self.last_entry_price is not None:
            required_price = self.last_entry_price * (1.0 - (self.p.min_add_drop_pct / 100.0))
            if self.trade_data.close[0] >= required_price:
                return False

        if self.p.rsi_add_drop_pct > 0 and self.last_entry_rsi is not None:
            required_rsi = self.last_entry_rsi * (1.0 - (self.p.rsi_add_drop_pct / 100.0))
            if self.rsi[0] > required_rsi:
                return False

        return True

    def _can_add_short(self) -> bool:
        if self.entry_count <= 0:
            return True

        if self.p.min_add_minutes > 0 and self.last_short_entry_dt is not None:
            now_dt = self.trade_data.datetime.datetime(0)
            elapsed_mins = (now_dt - self.last_short_entry_dt).total_seconds() / 60.0
            if elapsed_mins < self.p.min_add_minutes:
                return False

        if self.p.add_only_lower_price and self.last_short_entry_price is not None:
            required_price = self.last_short_entry_price * (1.0 + (self.p.min_add_drop_pct / 100.0))
            if self.trade_data.close[0] <= required_price:
                return False

        if self.p.rsi_add_drop_pct > 0 and self.last_short_entry_rsi is not None:
            required_rsi = self.last_short_entry_rsi * (1.0 + (self.p.rsi_add_drop_pct / 100.0))
            current_short_rsi = self._short_rsi_ref_value()
            if current_short_rsi is None or current_short_rsi < required_rsi:
                return False

        return True

    def next(self) -> None:
        if self.order:
            return

        self._update_ema_state()
        self._update_short_ema_state()
        long_condition = self._ema_ok() and self._rsi_ok() and self._volume_ok()
        short_condition = self._short_ema_ok() and self._short_rsi_ok() and self._volume_ok()

        # Long entries
        if self.p.enable_longs and long_condition and self.position.size >= 0 and self.entry_count < self.p.max_entries and self._can_add_entry():
            self.entry_signal_count += 1
            size = self._entry_size()
            if size > 0:
                self.order = self.buy(size=size)

        # Short entries
        if (not self.order) and self.p.enable_shorts and short_condition and self.position.size <= 0 and self.entry_count < self.p.max_entries and self._can_add_short():
            self.entry_signal_count += 1
            size = self._entry_size()
            if size > 0:
                self.order = self.sell(size=size)

        # Pine-like trailing logic on total open position
        if self.position.size > 0:
            self.cycle_ref_size = max(self.cycle_ref_size, self.position.size)
            position_value = self.position.size * self.trade_data.close[0]
            avg_value = self.position.size * self.position.price
            profit_usdt = position_value - avg_value

            self.max_profit_usdt = max(self.max_profit_usdt, profit_usdt)

            if (not self.trailing_active) and profit_usdt >= self.p.trail_activation_usdt:
                self.trailing_active = True
                self.stoploss_usdt = 0.0
                if self.p.use_tp_ladder and self.p.tp_levels > 0:
                    self.next_tp_target_usdt = self.p.trail_activation_usdt

            if self.trailing_active:
                base_profit = max(self.max_profit_usdt - self.p.trail_activation_offset_usdt, 0.0)
                new_sl = base_profit * self.p.trail_keep_pct
                self.stoploss_usdt = max(self.stoploss_usdt, new_sl)

            if (
                self.p.use_tp_ladder
                and self.trailing_active
                and self.p.tp_levels > 0
                and self.tp_hits < self.p.tp_levels
                and self.next_tp_target_usdt is not None
                and self.max_profit_usdt >= self.next_tp_target_usdt
            ):
                chunk_size = self.cycle_ref_size * (self.p.tp_close_pct / 100.0)
                chunk_size = min(chunk_size, self.position.size)
                if chunk_size > 0:
                    self.order = self.close(size=chunk_size)
                    self.tp_hits += 1
                    self.next_tp_target_usdt *= (1.0 + self.p.tp_step_pct / 100.0)
                    return

            if profit_usdt <= self.stoploss_usdt:
                self.order = self.close()
        elif self.position.size < 0:
            self.cycle_ref_size = max(self.cycle_ref_size, abs(self.position.size))
            position_value = abs(self.position.size) * self.trade_data.close[0]
            avg_value = abs(self.position.size) * self.position.price
            profit_usdt = avg_value - position_value

            self.max_profit_usdt = max(self.max_profit_usdt, profit_usdt)

            if (not self.trailing_active) and profit_usdt >= self.p.trail_activation_usdt:
                self.trailing_active = True
                self.stoploss_usdt = 0.0
                if self.p.use_short_tp_ladder and self.p.tp_levels > 0:
                    self.next_tp_target_usdt = self.p.trail_activation_usdt

            if self.trailing_active:
                base_profit = max(self.max_profit_usdt - self.p.trail_activation_offset_usdt, 0.0)
                new_sl = base_profit * self.p.trail_keep_pct
                self.stoploss_usdt = max(self.stoploss_usdt, new_sl)

            if (
                self.p.use_short_tp_ladder
                and self.trailing_active
                and self.p.tp_levels > 0
                and self.tp_hits < self.p.tp_levels
                and self.next_tp_target_usdt is not None
                and self.max_profit_usdt >= self.next_tp_target_usdt
            ):
                chunk_size = self.cycle_ref_size * (self.p.tp_close_pct / 100.0)
                chunk_size = min(chunk_size, abs(self.position.size))
                if chunk_size > 0:
                    self.order = self.close(size=chunk_size)
                    self.tp_hits += 1
                    self.next_tp_target_usdt *= (1.0 + self.p.tp_step_pct / 100.0)
                    return

            if profit_usdt <= self.stoploss_usdt:
                self.order = self.close()
        else:
            self.max_profit_usdt = 0.0
            self.stoploss_usdt = self.p.initial_stoploss_usdt
            self.trailing_active = False
            self.entry_count = 0
            self.last_entry_dt = None
            self.last_entry_price = None
            self.last_entry_rsi = None
            self.last_short_entry_dt = None
            self.last_short_entry_price = None
            self.last_short_entry_rsi = None
            self.tp_hits = 0
            self.next_tp_target_usdt = None
            self.cycle_ref_size = 0.0

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            now_dt = bt.num2date(order.executed.dt)
            if order.isbuy() and self.position.size > 0:
                self.entry_count += 1
                self.last_entry_dt = now_dt
                self.last_entry_price = order.executed.price
                self.last_entry_rsi = float(self.rsi[0])
                if self.entry_count == 1 and not self.trailing_active:
                    self.stoploss_usdt = self.p.initial_stoploss_usdt
            elif order.issell() and self.position.size < 0:
                self.entry_count += 1
                self.last_short_entry_dt = now_dt
                self.last_short_entry_price = order.executed.price
                self.last_short_entry_rsi = self._short_rsi_ref_value()
                if self.entry_count == 1 and not self.trailing_active:
                    self.stoploss_usdt = self.p.short_initial_stoploss_usdt
            if self.p.verbose_trades:
                print(
                    f"ORDER {side} @ {self._current_dt(order.data)} price={order.executed.price:.3f} size={order.executed.size:.4f} entries={self.entry_count}"
                )

        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.rejected_orders += 1
            if self.p.verbose_trades:
                print(f"ORDER REJECTED @ {self._current_dt(order.data)} status={order.getstatusname()}")

        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.trade_count += 1
            if trade.pnlcomm > 0:
                self.winning_trades += 1
            if trade.long:
                self.long_trade_count += 1
                self.long_pnl += trade.pnlcomm
            else:
                self.short_trade_count += 1
                self.short_pnl += trade.pnlcomm
            open_dt = bt.num2date(trade.dtopen).strftime("%Y-%m-%d %H:%M:%S")
            close_dt = bt.num2date(trade.dtclose).strftime("%Y-%m-%d %H:%M:%S")
            if self.first_trade_open_dt is None:
                self.first_trade_open_dt = open_dt
            self.last_trade_close_dt = close_dt
            if self.p.verbose_trades:
                print(
                    f"TRADE #{self.trade_count}: open={open_dt} close={close_dt} "
                    f"gross={trade.pnl:.2f} net={trade.pnlcomm:.2f}"
                )


class XAUUSDGenericCSVData(bt.feeds.GenericCSVData):
    params = (
        ("dtformat", "%Y%m%d"),
        ("tmformat", "%H%M%S"),
        ("datetime", 1),
        ("time", 2),
        ("open", 3),
        ("high", 4),
        ("low", 5),
        ("close", 6),
        ("volume", 7),
        ("openinterest", -1),
        ("nullvalue", 0.0),
        ("headers", False),
    )



def _parse_csv_datetime(raw_line: str) -> datetime:
    parts = raw_line.split(",")
    if len(parts) < 3:
        raise ValueError(f"Invalid data line: {raw_line}")
    return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")


def _period_metrics(start_value: float, end_value: float, start_dt: datetime, end_dt: datetime) -> tuple[float, float, float]:
    roi_pct = ((end_value / start_value) - 1.0) * 100.0 if start_value > 0 else 0.0
    period_days = max((end_dt - start_dt).total_seconds() / 86400.0, 1e-9)
    period_months = period_days / 30.4375
    annualized_roi_pct = ((end_value / start_value) ** (365.0 / period_days) - 1.0) * 100.0 if start_value > 0 else 0.0
    return roi_pct, period_months, annualized_roi_pct

def tail_to_tempfile(path: Path, tail_rows: int) -> tuple[Path, int, datetime, datetime, float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    lines = deque(maxlen=tail_rows)
    with path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if line:
                lines.append(line)

    if not lines:
        raise ValueError("Data file is empty after filtering blank lines")

    tmp = tempfile.NamedTemporaryFile(prefix="xauusd_tail_", suffix=".csv", mode="w", delete=False)
    with tmp:
        tmp.write("\n".join(lines))
        tmp.write("\n")

    start_dt = _parse_csv_datetime(lines[0])
    end_dt = _parse_csv_datetime(lines[-1])
    start_close = float(lines[0].split(",")[6])
    end_close = float(lines[-1].split(",")[6])
    return Path(tmp.name), len(lines), start_dt, end_dt, start_close, end_close


def _validate_tf(name: str, tf: int) -> None:
    if tf < 1:
        raise ValueError(f"{name} must be >= 1 minute")


def _parse_short_rsi_filters(value: str) -> list[dict[str, float | int]]:
    filters: list[dict[str, float | int]] = []
    if not value:
        return filters
    for raw in value.split(";"):
        token = raw.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                "short-rsi-filters entries must be 'tf:len:min:max' and separated by ';'"
            )
        tf, length = int(parts[0]), int(parts[1])
        min_val, max_val = float(parts[2]), float(parts[3])
        if tf < 1 or length < 1:
            raise ValueError("short-rsi-filters tf/len must be >= 1")
        if min_val >= max_val:
            raise ValueError("short-rsi-filters require min < max")
        filters.append({"tf": tf, "len": length, "min": min_val, "max": max_val})
    return filters


def run_backtest(args: argparse.Namespace) -> None:
    short_rsi_filters = _parse_short_rsi_filters(args.short_rsi_filters)
    _validate_tf("ema-tf", args.ema_tf)
    _validate_tf("rsi-tf", args.rsi_tf)
    _validate_tf("vol-tf", args.vol_tf)
    _validate_tf("short-ema-tf", args.short_ema_tf)
    _validate_tf("short-rsi-tf", args.short_rsi_tf)
    for idx, srf in enumerate(short_rsi_filters, start=1):
        _validate_tf(f"short-rsi-filters[{idx}].tf", int(srf["tf"]))
    if args.max_entries < 1:
        raise ValueError("max-entries must be >= 1")
    if not (0 < args.entry_cash_buffer <= 1):
        raise ValueError("entry-cash-buffer must be in (0, 1]")
    if args.min_add_minutes < 0:
        raise ValueError("min-add-minutes must be >= 0")
    if args.ema_min_above_hours < 0:
        raise ValueError("ema-min-above-hours must be >= 0")
    if args.min_add_drop_pct < 0:
        raise ValueError("min-add-drop-pct must be >= 0")
    if args.rsi_add_drop_pct < 0:
        raise ValueError("rsi-add-drop-pct must be >= 0")
    if args.trail_activation_offset_usdt < 0:
        raise ValueError("trail-activation-offset-usdt must be >= 0")
    if args.tp_levels < 0:
        raise ValueError("tp-levels must be >= 0")
    if args.tp_step_pct <= 0:
        raise ValueError("tp-step-pct must be > 0")
    if args.tp_close_pct <= 0:
        raise ValueError("tp-close-pct must be > 0")
    if args.use_tp_ladder:
        total_close_pct = args.tp_levels * args.tp_close_pct
        if total_close_pct > 100.0:
            raise ValueError("tp-levels * tp-close-pct must be <= 100")
        if total_close_pct < 50.0:
            raise ValueError("tp-levels * tp-close-pct must be >= 50 so runner is at most 50%")
    if args.use_short_tp_ladder:
        total_close_pct = args.tp_levels * args.tp_close_pct
        if total_close_pct > 100.0:
            raise ValueError("tp-levels * tp-close-pct must be <= 100")
        if total_close_pct < 50.0:
            raise ValueError("tp-levels * tp-close-pct must be >= 50 so runner is at most 50%")

    tail_path, loaded_rows, start_dt, end_dt, start_close, end_close = tail_to_tempfile(args.data, tail_rows=args.tail)

    minimum_rows = max(args.ema_period + 1, args.rsi_len + 1, args.vol_len + 1)
    if loaded_rows < minimum_rows:
        raise ValueError(
            f"Not enough rows for configured indicators. Need at least {minimum_rows}, got {loaded_rows}."
        )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(args.cash)

    base_data = XAUUSDGenericCSVData(dataname=str(tail_path), timeframe=bt.TimeFrame.Minutes, compression=1)
    base_data.plotinfo.plotvolume = False
    cerebro.adddata(base_data)

    tf_map = {1: 0}
    next_idx = 1
    for tf in sorted({
        args.ema_tf,
        args.rsi_tf,
        args.vol_tf,
        args.short_ema_tf,
        args.short_rsi_tf,
        *[int(cfg["tf"]) for cfg in short_rsi_filters],
    }):
        if tf == 1:
            continue
        resampled = cerebro.resampledata(base_data, timeframe=bt.TimeFrame.Minutes, compression=tf)
        resampled.plotinfo.plot = False
        tf_map[tf] = next_idx
        next_idx += 1

    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Trades)

    cerebro.addstrategy(
        Step1SimpleStrategy,
        enable_longs=args.enable_longs,
        enable_shorts=args.enable_shorts,
        use_ema_filter=args.use_ema,
        ema_period=args.ema_period,
        ema_condition=args.ema_condition,
        ema_min_above_hours=args.ema_min_above_hours,
        use_rsi=args.use_rsi,
        rsi_len=args.rsi_len,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        use_volume=args.use_volume,
        vol_len=args.vol_len,
        vol_multiplier=args.vol_multiplier,
        entry_cash=args.entry_cash,
        entry_cash_buffer=args.entry_cash_buffer,
        max_entries=args.max_entries,
        min_add_minutes=args.min_add_minutes,
        add_only_lower_price=args.add_only_lower_price,
        min_add_drop_pct=args.min_add_drop_pct,
        rsi_add_drop_pct=args.rsi_add_drop_pct,
        trail_activation_usdt=args.trail_activation_usdt,
        initial_stoploss_usdt=args.initial_stoploss_usdt,
        short_initial_stoploss_usdt=args.short_initial_stoploss_usdt,
        trail_keep_pct=args.trail_keep_pct,
        trail_activation_offset_usdt=args.trail_activation_offset_usdt,
        use_tp_ladder=args.use_tp_ladder,
        use_short_tp_ladder=args.use_short_tp_ladder,
        tp_levels=args.tp_levels,
        tp_step_pct=args.tp_step_pct,
        tp_close_pct=args.tp_close_pct,
        verbose_trades=not args.quiet,
        ema_data_idx=tf_map[args.ema_tf],
        rsi_data_idx=tf_map[args.rsi_tf],
        vol_data_idx=tf_map[args.vol_tf],
        short_ema_data_idx=tf_map[args.short_ema_tf],
        short_rsi_data_idx=tf_map[args.short_rsi_tf],
        short_rsi_filters=short_rsi_filters,
        short_rsi_filter_data_idxs=[tf_map[int(cfg["tf"])] for cfg in short_rsi_filters],
    )

    print(f"Rows loaded: {loaded_rows}")
    print(
        "Data window: "
        f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} -> {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Start portfolio value: {cerebro.broker.getvalue():.2f}")
    strategies = cerebro.run()
    end_value = cerebro.broker.getvalue()
    print(f"End portfolio value: {end_value:.2f}")
    print(f"EMA enabled: {args.use_ema} (tf={args.ema_tf}m, cond={args.ema_condition}, min-above-h={args.ema_min_above_hours})")
    print(f"Short EMA enabled: {args.use_short_ema} (tf={args.short_ema_tf}m, cond={args.short_ema_condition}, min-below-h={args.short_ema_min_below_hours})")
    print(f"RSI enabled: {args.use_rsi} (tf={args.rsi_tf}m)")
    print(f"Short RSI enabled: {args.use_short_rsi} (tf={args.short_rsi_tf}m)")
    if short_rsi_filters:
        print(f"Short RSI filters (multi): {args.short_rsi_filters}")
    print(f"Volume enabled: {args.use_volume} (tf={args.vol_tf}m)")
    print(
        f"Entry cash={args.entry_cash:.2f}, cash buffer={args.entry_cash_buffer:.2f}, max entries={args.max_entries}, "
        f"min add wait={args.min_add_minutes}m, add-only-lower={args.add_only_lower_price}, "
        f"min add drop={args.min_add_drop_pct:.2f}%, rsi add drop={args.rsi_add_drop_pct:.2f}%, "
        f"trail activation={args.trail_activation_usdt:.2f}, trail offset={args.trail_activation_offset_usdt:.2f}, initial SL long={args.initial_stoploss_usdt:.2f}, initial SL short={args.short_initial_stoploss_usdt:.2f}, keep={args.trail_keep_pct:.2f}, "
        f"tp ladder={args.use_tp_ladder}, tp levels={args.tp_levels}, tp step={args.tp_step_pct:.2f}%, tp close={args.tp_close_pct:.2f}%"
    )
    print(f"Entry signals: {strategies[0].entry_signal_count}")
    print(f"Rejected/Margin orders: {strategies[0].rejected_orders}")
    print(f"Closed trades: {strategies[0].trade_count}")
    print(f"Long closed trades: {strategies[0].long_trade_count}")
    print(f"Short closed trades: {strategies[0].short_trade_count}")
    print(f"Long PnL (net): {strategies[0].long_pnl:.2f}")
    print(f"Short PnL (net): {strategies[0].short_pnl:.2f}")
    print(
        "Trade window: "
        f"{strategies[0].first_trade_open_dt or 'n/a'} -> "
        f"{strategies[0].last_trade_close_dt or 'n/a'}"
    )
    winrate = (strategies[0].winning_trades / strategies[0].trade_count * 100.0) if strategies[0].trade_count else 0.0
    print(f"Win rate: {winrate:.2f}%")
    roi_pct, period_months, annualized_roi_pct = _period_metrics(args.cash, end_value, start_dt, end_dt)
    print(f"ROI period: {roi_pct:.2f}% over {period_months:.1f} months")
    print(f"Annualized ROI: {annualized_roi_pct:.2f}%")
    hodl_end_value = args.cash * (end_close / start_close) if start_close > 0 else args.cash
    hodl_roi_pct = ((hodl_end_value / args.cash) - 1.0) * 100.0 if args.cash > 0 else 0.0
    print(f"HODL ROI period: {hodl_roi_pct:.2f}% (end value {hodl_end_value:.2f})")

    if args.plot:
        try:
            cerebro.plot(style="candlestick", volume=False)
        except ImportError as exc:
            print(f"Plot skipped: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step-1 XAUUSD backtest")
    parser.add_argument("--data", type=Path, default=Path("XAUUSD.txt"), help="Path to XAUUSD data file")
    parser.add_argument("--cash", type=float, default=1500.0, help="Starting cash")
    parser.add_argument("--tail", type=int, default=30000, help="Rows to keep from tail")

    parser.add_argument("--enable-longs", dest="enable_longs", action="store_true", help="Enable long entries")
    parser.add_argument("--disable-longs", dest="enable_longs", action="store_false", help="Disable long entries")
    parser.set_defaults(enable_longs=True)
    parser.add_argument("--enable-shorts", dest="enable_shorts", action="store_true", help="Enable short entries")
    parser.add_argument("--disable-shorts", dest="enable_shorts", action="store_false", help="Disable short entries")
    parser.set_defaults(enable_shorts=False)

    parser.add_argument("--entry-cash", type=float, default=500.0, help="Cash used per entry (Pine default_qty_value)")
    parser.add_argument("--entry-cash-buffer", type=float, default=0.98, help="Safety buffer for entry cash to reduce margin rejects")
    parser.add_argument("--max-entries", type=int, default=3, help="Maximum add-on entries while in position (pyramiding)")
    parser.add_argument("--min-add-minutes", type=int, default=5, help="Minimum minutes before another add-on entry")
    parser.add_argument("--add-only-lower-price", dest="add_only_lower_price", action="store_true", help="Allow add-ons only below last long entry / above last short entry")
    parser.add_argument("--no-add-only-lower-price", dest="add_only_lower_price", action="store_false", help="Disable directional add-on price rule")
    parser.set_defaults(add_only_lower_price=True)
    parser.add_argument("--min-add-drop-pct", type=float, default=0.0, help="Require add-on price move by at least X%% from last directional entry")
    parser.add_argument("--rsi-add-drop-pct", type=float, default=0.0, help="Require directional RSI move by at least X%% from last directional entry")

    parser.add_argument("--trail-activation-usdt", type=float, default=4.0, help="Profit in USDT to activate trailing")
    parser.add_argument("--initial-stoploss-usdt", type=float, default=-20.0, help="Initial USDT stoploss before trailing")
    parser.add_argument("--short-initial-stoploss-usdt", type=float, default=-20.0, help="Initial USDT stoploss for short positions before trailing")
    parser.add_argument("--trail-keep-pct", type=float, default=0.5, help="Trailing keep percentage of max profit")
    parser.add_argument("--trail-activation-offset-usdt", type=float, default=0.0, help="Subtract X USDT from max profit before applying trail keep pct")

    parser.add_argument("--use-tp-ladder", dest="use_tp_ladder", action="store_true", help="Enable multi-level partial take-profits for longs")
    parser.add_argument("--no-use-tp-ladder", dest="use_tp_ladder", action="store_false", help="Disable TP ladder for longs")
    parser.set_defaults(use_tp_ladder=False)
    parser.add_argument("--use-short-tp-ladder", dest="use_short_tp_ladder", action="store_true", help="Enable multi-level partial take-profits for shorts")
    parser.add_argument("--no-use-short-tp-ladder", dest="use_short_tp_ladder", action="store_false", help="Disable TP ladder for shorts")
    parser.set_defaults(use_short_tp_ladder=False)
    parser.add_argument("--tp-levels", type=int, default=0, help="Number of TP levels")
    parser.add_argument("--tp-step-pct", type=float, default=10.0, help="Percent spacing between TP levels (on max profit target)")
    parser.add_argument("--tp-close-pct", type=float, default=10.0, help="Percent of reference position to close per TP level")

    parser.add_argument("--use-ema", dest="use_ema", action="store_true", help="Enable long EMA filter")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false", help="Disable long EMA filter")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--ema-period", type=int, default=50, help="Long EMA period")
    parser.add_argument("--ema-condition", choices=["above", "below", "below_after_above"], default="below_after_above", help="Long EMA condition")
    parser.add_argument("--ema-min-above-hours", type=float, default=2.0, help="Required hours above EMA before below signal activates")
    parser.add_argument("--ema-tf", type=int, default=15, help="Long EMA timeframe in minutes")

    parser.add_argument("--use-short-ema", dest="use_short_ema", action="store_true", help="Enable short EMA filter")
    parser.add_argument("--no-use-short-ema", dest="use_short_ema", action="store_false", help="Disable short EMA filter")
    parser.set_defaults(use_short_ema=True)
    parser.add_argument("--short-ema-period", type=int, default=50, help="Short EMA period")
    parser.add_argument("--short-ema-condition", choices=["above", "below", "above_after_below"], default="above_after_below", help="Short EMA condition")
    parser.add_argument("--short-ema-min-below-hours", type=float, default=2.0, help="Required hours below short EMA before above signal activates")
    parser.add_argument("--short-ema-tf", type=int, default=15, help="Short EMA timeframe in minutes")

    parser.add_argument("--use-rsi", action="store_true", default=False, help="Enable long RSI filter")
    parser.add_argument("--rsi-len", type=int, default=50, help="Long RSI length")
    parser.add_argument("--rsi-min", type=float, default=20.0, help="Long RSI lower bound")
    parser.add_argument("--rsi-max", type=float, default=80.0, help="Long RSI upper bound")
    parser.add_argument("--rsi-tf", type=int, default=1, help="Long RSI timeframe in minutes")

    parser.add_argument("--use-short-rsi", action="store_true", default=False, help="Enable short RSI filter")
    parser.add_argument("--short-rsi-len", type=int, default=50, help="Short RSI length")
    parser.add_argument("--short-rsi-min", type=float, default=20.0, help="Short RSI lower bound")
    parser.add_argument("--short-rsi-max", type=float, default=80.0, help="Short RSI upper bound")
    parser.add_argument("--short-rsi-tf", type=int, default=1, help="Short RSI timeframe in minutes")
    parser.add_argument(
        "--short-rsi-filters",
        type=str,
        default="",
        help="Optional multi short RSI filters as 'tf:len:min:max;tf:len:min:max'",
    )

    parser.add_argument("--use-volume", action="store_true", default=False, help="Enable volume filter")
    parser.add_argument("--vol-len", type=int, default=50, help="Volume SMA length")
    parser.add_argument("--vol-multiplier", type=float, default=1.5, help="Volume threshold multiplier")
    parser.add_argument("--vol-tf", type=int, default=1, help="Volume filter timeframe in minutes")

    parser.add_argument("--quiet", action="store_true", help="Suppress per-order/per-trade logs and print only summary")
    parser.add_argument("--plot", action="store_true", help="Show backtrader chart with price/equity")

    return parser.parse_args()


if __name__ == "__main__":
    run_backtest(parse_args())
