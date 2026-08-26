# -*- coding: utf-8 -*-
"""采集不可稳定回补的 AKShare A 股市场状态快照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


PROTOCOL_VERSION = 1
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def now_iso() -> str:
    """返回带本地时区偏移的 ISO 时间。"""
    return datetime.now().astimezone().isoformat()


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """在同目录原子写入 JSON，避免消费者读取半成品。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".partial_", suffix=".json", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def sha256_file(path: str) -> str:
    """流式计算结果文件的 SHA256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(
    path: str,
    dataset: str,
    rows: int = 0,
    min_time: Any = None,
    max_time: Any = None,
) -> Dict[str, Any]:
    """生成与 TDX 数据湖兼容的文件清单项。"""
    return {
        "path": os.path.basename(path),
        "dataset": dataset,
        "sha256": sha256_file(path),
        "size": os.path.getsize(path),
        "rows": int(rows or 0),
        "min_time": min_time,
        "max_time": max_time,
    }


def collect_market_snapshot(ak_module=None, snapshot_time=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """调用 AKShare 并返回市场汇总和行业资金流标准表。"""
    if ak_module is None:
        try:
            import akshare as ak_module
        except ImportError as exc:
            raise RuntimeError("当前 Python 环境缺少 akshare，请先安装依赖") from exc
    timestamp = pd.Timestamp(snapshot_time or pd.Timestamp.now(tz="Asia/Shanghai"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    timestamp_text = timestamp.isoformat()
    errors = []
    spot, spot_source = _collect_spot(ak_module, errors)
    market_summary = _collect_market_summary(ak_module, errors)
    high_low = _safe_call(ak_module, "stock_a_high_low_statistics", errors=errors)
    trading_calendar = _safe_call(
        ak_module, "tool_trade_date_hist_sina", errors=errors)
    trade_date = _resolve_trade_date(timestamp, high_low, trading_calendar)
    date_text = trade_date.replace("-", "")
    limit_up = _safe_call(ak_module, "stock_zt_pool_em", errors=errors, date=date_text)
    limit_down = _safe_call(ak_module, "stock_zt_pool_dtgc_em", errors=errors, date=date_text)
    broken = _safe_call(ak_module, "stock_zt_pool_zbgc_em", errors=errors, date=date_text)
    sectors, sector_source = _collect_sectors(ak_module, errors)

    change = _numeric_column(spot, ("涨跌幅", "涨跌幅%", "changepercent"))
    amount = _numeric_column(spot, ("成交额", "amount"))
    up_count = int((change > 0).sum())
    down_count = int((change < 0).sum())
    flat_count = int((change == 0).sum())
    universe_count = int(change.notna().sum())
    if market_summary:
        up_count = int(market_summary["up_count"])
        down_count = int(market_summary["down_count"])
        flat_count = int(market_summary["flat_count"])
        universe_count = int(market_summary["universe_count"])
    st_limits = _collect_st_limit_supplement(ak_module, errors) if not market_summary else {}
    limit_up_count = int(
        market_summary.get("limit_up_count", len(limit_up) + st_limits.get("limit_up_count", 0)))
    limit_down_count = int(
        market_summary.get("limit_down_count", len(limit_down) + st_limits.get("limit_down_count", 0)))
    metrics: Dict[str, Any] = {
        "schema_version": 1,
        "snapshot_time": timestamp_text,
        "trade_date": trade_date,
        "universe_count": universe_count,
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "up_ratio": up_count / universe_count if universe_count else np.nan,
        "change_mean": change.mean(),
        "change_median": change.median(),
        "change_q10": change.quantile(0.10),
        "change_q90": change.quantile(0.90),
        "total_amount": amount.sum(min_count=1),
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "broken_limit_count": len(broken),
        "seal_success_ratio": (
            limit_up_count / float(limit_up_count + len(broken))
            if limit_up_count + len(broken) else np.nan
        ),
        "max_limit_up_streak": _numeric_column(limit_up, ("连板数",)).max(),
        "spot_source": spot_source,
        "breadth_source": "ths" if market_summary else spot_source,
        "limit_source": (
            "ths" if market_summary
            else "eastmoney_pool+legu_st" if st_limits
            else "eastmoney_pool"
        ),
        "sector_source": sector_source,
        "quality_warnings": " | ".join(errors),
        "source": "akshare",
        "quality_status": "complete" if universe_count else "failed",
    }
    for field in (
            "breadth_distribution", "limit_up_intraday_high",
            "limit_down_intraday_high", "previous_limit_up_return", "market_rating"):
        if field in market_summary:
            metrics[field] = market_summary[field]
    if high_low is not None and not high_low.empty:
        latest = _high_low_row(high_low, trade_date)
        for output, candidates in {
            "high_20_count": ("20日新高", "20日新高家数", "high20"),
            "low_20_count": ("20日新低", "20日新低家数", "low20"),
            "high_60_count": ("60日新高", "60日新高家数", "high60"),
            "low_60_count": ("60日新低", "60日新低家数", "low60"),
            "high_120_count": ("120日新高", "120日新高家数", "high120"),
            "low_120_count": ("120日新低", "120日新低家数", "low120"),
        }.items():
            metrics[output] = _row_numeric(latest, candidates)
    state = pd.DataFrame([metrics])
    if not sectors.empty:
        sectors["snapshot_time"] = timestamp_text
        sectors["trade_date"] = trade_date
        sectors["source"] = "akshare"
        sectors["provider_source"] = sector_source
        sectors["quality_status"] = metrics["quality_status"]
    return state, sectors


def publish_snapshot(
    exchange_root: str | Path,
    ak_module=None,
    snapshot_time=None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """把快照作为不可重放结果包原子发布到 ready 目录。"""
    state, sectors = collect_market_snapshot(ak_module=ak_module, snapshot_time=snapshot_time)
    timestamp = pd.Timestamp(state.iloc[0]["snapshot_time"])
    job_id = job_id or "akshare-%s-%s" % (
        timestamp.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8])
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("非法 job_id")
    root = Path(exchange_root).expanduser().resolve()
    ready = root / "ready"
    temporary = ready / ("." + job_id + ".partial")
    final = ready / job_id
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        files = []
        for dataset, frame in (
            ("akshare_market_state", state),
            ("akshare_sector_flow", sectors),
        ):
            if frame.empty:
                continue
            path = temporary / (dataset + ".csv.gz")
            frame.to_csv(path, index=False, encoding="utf-8", compression="gzip")
            files.append(build_file_manifest(
                str(path), dataset, len(frame),
                frame["snapshot_time"].min(), frame["snapshot_time"].max()))
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "status": "completed",
            "source": "akshare",
            "created_at": now_iso(),
            "completed_at": now_iso(),
            "request": {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job_id,
                "job_type": "akshare_snapshot_export",
                "scope": "trade_date",
                "symbols": [],
                "frequencies": [],
                "created_at": now_iso(),
            },
            "files": files,
            "warnings": [state.iloc[0]["quality_warnings"]]
            if state.iloc[0]["quality_warnings"] else [],
            "replayable": False,
        }
        atomic_write_json(str(temporary / "manifest.json"), manifest)
        os.replace(str(temporary), str(final))
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _collect_spot(ak_module, errors) -> Tuple[pd.DataFrame, str]:
    """优先使用东方财富，失败时降级到新浪全市场现货。"""
    frame = _safe_call(ak_module, "stock_zh_a_spot_em", errors=errors)
    if not frame.empty:
        return frame, "eastmoney"
    frame = _safe_call(ak_module, "stock_zh_a_spot", errors=errors)
    return frame, "sina" if not frame.empty else "unavailable"


def _collect_market_summary(ak_module, errors) -> Dict[str, Any]:
    """读取同花顺大盘汇总，统一市场宽度和含 ST 的涨跌停口径。"""
    injected = getattr(ak_module, "stock_market_summary_ths", None)
    if injected is not None:
        try:
            return dict(injected())
        except Exception as exc:
            errors.append("stock_market_summary_ths: %s" % exc)
            return {}
    if getattr(ak_module, "__name__", "") != "akshare":
        return {}
    try:
        import py_mini_racer
        import requests
        from akshare.stock_feature.stock_fund_flow import _get_file_content_ths

        engine = py_mini_racer.MiniRacer()
        engine.eval(_get_file_content_ths("ths.js"))
        headers = {
            "hexin-v": engine.call("v"),
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/90.0.4430.85 Safari/537.36"
            ),
            "Referer": "http://q.10jqka.com.cn/",
            "X-Requested-With": "XMLHttpRequest",
        }
        response = requests.get(
            "https://q.10jqka.com.cn/api.php?t=indexflash&",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        breadth = payload["zdfb_data"]
        limits = payload["zdt_data"]["last_zdt"]
        up_count = int(breadth["znum"])
        down_count = int(breadth["dnum"])
        universe_count = int(sum(int(value) for value in breadth["zdfb"]))
        result = {
            "universe_count": universe_count,
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": max(0, universe_count - up_count - down_count),
            "limit_up_count": int(limits["ztzs"]),
            "limit_down_count": int(limits["dtzs"]),
            "breadth_distribution": json.dumps(
                [int(value) for value in breadth["zdfb"]], separators=(",", ":")),
        }
        intraday = payload.get("zdt_data") or {}
        if intraday.get("ztzs"):
            result["limit_up_intraday_high"] = int(max(intraday["ztzs"]))
        if intraday.get("dtzs"):
            result["limit_down_intraday_high"] = int(max(intraday["dtzs"]))
        previous_limit = (payload.get("jrbx_data") or {}).get("last_zdf")
        if previous_limit is not None:
            result["previous_limit_up_return"] = float(previous_limit)
        market_rating = payload.get("dppj_data")
        if market_rating is not None:
            result["market_rating"] = float(market_rating)
        return result
    except Exception as exc:
        errors.append("stock_market_summary_ths: %s" % exc)
        return {}


def _collect_st_limit_supplement(ak_module, errors) -> Dict[str, int]:
    """同花顺汇总不可用时，用乐咕补齐东方财富普通股池遗漏的 ST。"""
    frame = _safe_call(ak_module, "stock_market_activity_legu", errors=errors)
    if frame.empty or "item" not in frame.columns or "value" not in frame.columns:
        return {}
    values = {
        str(row["item"]).strip().lower().replace(" ", ""): row["value"]
        for _, row in frame.iterrows()
    }

    def parse_count(name):
        """安全解析乐咕表中的计数值。"""
        value = pd.to_numeric(pd.Series([values.get(name)]), errors="coerce").iloc[0]
        return int(value) if not pd.isna(value) else 0

    return {
        "limit_up_count": parse_count("stst*涨停"),
        "limit_down_count": parse_count("stst*跌停"),
    }


def _collect_sectors(ak_module, errors) -> Tuple[pd.DataFrame, str]:
    """采集行业主力资金排名并统一字段，同花顺失败时降级到东方财富。"""
    frame = _safe_call(
        ak_module, "stock_fund_flow_industry", errors=errors, symbol="即时")
    source = "ths"
    amount_candidates = ("净额",)
    if frame.empty:
        frame = _safe_call(
            ak_module, "stock_sector_fund_flow_rank", errors=errors,
            indicator="今日", sector_type="行业资金流")
        source = "eastmoney"
        amount_candidates = ("今日主力净流入-净额", "主力净流入-净额", "主力净流入")
    if frame.empty:
        return pd.DataFrame(
            columns=["sector_name", "change_pct", "main_net_inflow", "rank"]), "unavailable"
    result = pd.DataFrame({
        "sector_name": _text_column(frame, ("名称", "行业", "板块名称")),
        "change_pct": _numeric_column(frame, ("今日涨跌幅", "行业-涨跌幅", "涨跌幅")),
        "main_net_inflow": _numeric_column(frame, amount_candidates),
    })
    if source == "ths":
        result["main_net_inflow"] = result["main_net_inflow"] * 100_000_000
    result = result.dropna(subset=["sector_name"])
    result["rank"] = range(1, len(result) + 1)
    return result, source


def _resolve_trade_date(timestamp: pd.Timestamp, high_low: pd.DataFrame,
                        trading_calendar: Optional[pd.DataFrame] = None) -> str:
    """结合交易日历和收盘时间确定快照所属交易日。"""
    local_time = timestamp.tz_convert("Asia/Shanghai") if timestamp.tzinfo else timestamp
    local_time = local_time.tz_localize(None) if local_time.tzinfo else local_time
    local_date = local_time.normalize()
    if trading_calendar is not None and not trading_calendar.empty:
        for column in ("trade_date", "date", "日期"):
            if column not in trading_calendar.columns:
                continue
            values = pd.to_datetime(
                trading_calendar[column], errors="coerce").dropna().dt.normalize()
            values = values[values <= local_date]
            if (local_time.hour, local_time.minute) < (9, 15):
                values = values[values < local_date]
            if not values.empty:
                return values.max().strftime("%Y-%m-%d")
    if local_date.weekday() < 5 and (local_time.hour, local_time.minute) >= (9, 15):
        # 交易时段及刚收盘时，部分历史统计仍停留于上一日，不能据此回退标签。
        return local_date.strftime("%Y-%m-%d")
    if high_low is not None and not high_low.empty:
        for column in ("date", "日期"):
            if column not in high_low.columns:
                continue
            values = pd.to_datetime(high_low[column], errors="coerce").dropna()
            values = values[values <= local_date]
            if not values.empty:
                return values.max().strftime("%Y-%m-%d")
    while local_date.weekday() >= 5:
        local_date -= pd.Timedelta(days=1)
    return local_date.strftime("%Y-%m-%d")


def _high_low_row(high_low: pd.DataFrame, trade_date: str) -> pd.Series:
    """读取目标交易日的新高新低统计，缺少日期列时沿用末行。"""
    for column in ("date", "日期"):
        if column not in high_low.columns:
            continue
        values = pd.to_datetime(high_low[column], errors="coerce")
        matched = high_low[values.dt.strftime("%Y-%m-%d") == trade_date]
        if not matched.empty:
            return matched.iloc[-1]
    return high_low.iloc[-1]


def _safe_call(module, name: str, errors=None, **kwargs) -> pd.DataFrame:
    """单个 AKShare 接口失败时返回空表，让其他指标仍可落盘。"""
    function = getattr(module, name, None)
    if function is None:
        return pd.DataFrame()
    try:
        result = function(**kwargs)
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
    except Exception as exc:
        if errors is not None:
            errors.append("%s: %s" % (name, exc))
        return pd.DataFrame()


def _numeric_column(frame: pd.DataFrame, candidates) -> pd.Series:
    """从候选中文字段中读取数值列。"""
    if frame is None:
        return pd.Series(dtype=float)
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _text_column(frame: pd.DataFrame, candidates) -> pd.Series:
    """从候选中文字段中读取文本列。"""
    for column in candidates:
        if column in frame.columns:
            return frame[column].astype(str).replace("nan", np.nan)
    return pd.Series(np.nan, index=frame.index, dtype=object)


def _row_numeric(row: pd.Series, candidates):
    """从一行中读取首个可用数值。"""
    for column in candidates:
        if column in row.index:
            return pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return np.nan


def main() -> None:
    """提供给独立采集后端任务服务调用的命令行入口。"""
    parser = argparse.ArgumentParser(description="采集 AKShare A 股市场状态快照")
    parser.add_argument("--exchange-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    manifest = publish_snapshot(args.exchange_root, job_id=args.job_id)
    print("AKShare snapshot ready: %s" % manifest["job_id"])


if __name__ == "__main__":
    main()
