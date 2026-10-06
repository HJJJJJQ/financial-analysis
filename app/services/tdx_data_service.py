from __future__ import annotations

import hmac
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Request

from app.config import DATA_DIR, ROOT_DIR
from app.services.job_service import get_job_state, start_command_job


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FILE_NAMES = {
    "manifest.json",
    "akshare_market_state.csv.gz",
    "akshare_sector_flow.csv.gz",
}
AKSHARE_OPERATIONS = frozenset({
    "fund_etf_hist_em",
    "stock_info_a_code_name",
    "stock_info_sz_change_name",
    "stock_lhb_detail_em",
    "stock_lhb_hyyyb_em",
    "stock_lhb_jgmmtj_em",
    "stock_lhb_stock_statistic_em",
    "stock_lhb_yybph_em",
    "stock_repurchase_em",
    "stock_zh_a_gdhs_detail_em",
    "stock_zh_a_hist",
    "stock_zh_a_hist_tx",
    "stock_zh_a_st_em",
    "stock_zt_pool_dtgc_em",
    "stock_zt_pool_em",
    "stock_zt_pool_zbgc_em",
    "tool_trade_date_hist_sina",
})


def export_root() -> Path:
    """返回 TDX 交换产物目录，默认位于被 Git 忽略的 data 下。"""
    configured = os.environ.get("FINANCIAL_ANALYSIS_TDX_EXPORT_DIR")
    return Path(configured or DATA_DIR / "tdx_exports").expanduser().resolve()


def authorize_machine_request(request: Request) -> bool:
    """配置 Token 时校验 Bearer 认证，未配置时保持本机兼容模式。"""
    expected = os.environ.get("FINANCIAL_ANALYSIS_API_TOKEN", "")
    if not expected:
        return True
    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        return False
    return hmac.compare_digest(header[len(prefix):], expected)


def start_market_snapshot() -> dict[str, Any]:
    """启动互斥的 AKShare 市场快照任务。"""
    job_id = "akshare-%s" % uuid.uuid4().hex
    root = export_root()
    root.mkdir(parents=True, exist_ok=True)
    started = start_command_job(
        job_id,
        [
            sys.executable,
            "-B",
            "-m",
            "app.services.tdx_market_snapshot",
            "--exchange-root",
            str(root),
            "--job-id",
            job_id,
        ],
        cwd=ROOT_DIR,
        timeout=900,
        resource_key="tdx-akshare-market-snapshot",
    )
    if not started:
        raise RuntimeError("已有 AKShare 市场快照任务正在运行")
    return machine_job_state(job_id)


def machine_job_state(job_id: str) -> dict[str, Any]:
    """返回机器任务状态，并在成功后附上 manifest 元数据。"""
    normalized = validate_job_id(job_id)
    state = get_job_state(normalized)
    manifest_path = artifact_path(normalized, "manifest.json")
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif not state.get("running") and state.get("started_at") is None:
        raise FileNotFoundError("任务不存在")
    payload = {
        "job_id": normalized,
        "running": bool(state.get("running")),
        "ok": True if manifest is not None else state.get("ok"),
        "error": state.get("error") or "",
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "elapsed_sec": state.get("elapsed_sec"),
    }
    if manifest is not None:
        payload["manifest"] = manifest
    return payload


def artifact_path(job_id: str, filename: str) -> Path:
    """解析受白名单限制的任务产物路径。"""
    normalized = validate_job_id(job_id)
    if filename not in _FILE_NAMES:
        raise ValueError("不支持的产物文件")
    package = (export_root() / "ready" / normalized).resolve()
    path = (package / filename).resolve()
    if path.parent != package:
        raise ValueError("非法产物路径")
    return path


def validate_job_id(job_id: str) -> str:
    """校验跨服务任务标识，阻止目录穿越。"""
    value = str(job_id or "")
    if not _JOB_ID_RE.fullmatch(value):
        raise ValueError("非法 job_id")
    return value


def call_akshare_operation(
    operation: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """执行白名单 AKShare 表接口并返回可移植的 split 结构。"""
    if operation not in AKSHARE_OPERATIONS:
        raise ValueError("不支持的 AKShare 操作")
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs 必须是对象")
    if any(
        not isinstance(key, str)
        or not isinstance(value, (str, int, float, bool, type(None)))
        for key, value in kwargs.items()
    ):
        raise ValueError("AKShare 参数只允许标量")
    if operation == "stock_repurchase_em":
        if kwargs:
            raise ValueError("stock_repurchase_em 不接受参数")
        from app.services.repurchase_source import call_repurchase_source
        return call_repurchase_source()
    import akshare

    function = getattr(akshare, operation, None)
    if not callable(function):
        raise RuntimeError("当前 AKShare 版本缺少接口: %s" % operation)
    result = function(**kwargs)
    frame = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
    payload = json.loads(frame.to_json(
        orient="split", date_format="iso", force_ascii=False))
    return {
        "operation": operation,
        "kind": "dataframe",
        "columns": payload.get("columns") or [],
        "data": payload.get("data") or [],
    }


def call_baostock_history(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """读取单只A股不复权历史成交额和换手率。"""
    normalized = str(symbol or "").strip().lower()
    if not re.fullmatch(r"(sh|sz|bj)\.\d{6}", normalized):
        raise ValueError("BaoStock 股票代码必须形如 sh.600000")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(start_date or "")):
        raise ValueError("BaoStock start_date 必须为 YYYY-MM-DD")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(end_date or "")):
        raise ValueError("BaoStock end_date 必须为 YYYY-MM-DD")

    try:
        import baostock
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境缺少 baostock，请先安装依赖") from exc

    login = baostock.login()
    if str(getattr(login, "error_code", "")) != "0":
        raise RuntimeError(
            "BaoStock 登录失败: %s" % getattr(login, "error_msg", "unknown"))
    try:
        result = baostock.query_history_k_data_plus(
            normalized,
            "date,code,volume,amount,turn,tradestatus",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        if str(getattr(result, "error_code", "")) != "0":
            raise RuntimeError(
                "BaoStock 历史行情失败: %s" % getattr(
                    result, "error_msg", "unknown"))
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=list(result.fields))
    finally:
        baostock.logout()

    payload = json.loads(frame.to_json(
        orient="split", date_format="iso", force_ascii=False))
    return {
        "operation": "query_history_k_data_plus",
        "kind": "dataframe",
        "columns": payload.get("columns") or [],
        "data": payload.get("data") or [],
    }
