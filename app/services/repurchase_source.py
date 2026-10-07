from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import requests


_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_COLUMN_MAP = {
    "DIM_SCODE": "股票代码", "SECURITYSHORTNAME": "股票简称", "NEWPRICE": "最新价",
    "REPURPRICECAP": "计划回购价格区间", "REPURNUMLOWER": "计划回购数量区间-下限",
    "REPURNUMCAP": "计划回购数量区间-上限", "ZSZXX": "占公告前一日总股本比例-下限",
    "ZSZSX": "占公告前一日总股本比例-上限", "JEXX": "计划回购金额区间-下限",
    "JESX": "计划回购金额区间-上限", "DIM_TRADEDATE": "回购起始时间",
    "REPURPROGRESS": "实施进度", "REPURPRICELOWER1": "已回购股份价格区间-下限",
    "REPURPRICECAP1": "已回购股份价格区间-上限", "REPURNUM": "已回购股份数量",
    "REPURAMOUNT": "已回购金额", "UPDATEDATE": "最新公告日期",
}
_PROGRESS = {"001": "董事会预案", "002": "股东大会通过", "003": "股东大会否决",
             "004": "实施中", "005": "停止实施", "006": "完成实施"}


class RepurchaseSourceError(RuntimeError):
    """保留来源失败和退避时间，不能将旧缓存当成刷新成功。"""

    def __init__(self, message: str, retry_after_seconds: float = 60):
        """构造可由机器接口返回的有界来源错误。"""
        self.retry_after_seconds = max(1, math.ceil(retry_after_seconds))
        super().__init__(message)


class RepurchaseSource:
    """串行取得完整回购快照，合并并发请求并短期复用同一源观察。"""

    def __init__(self, *, ttl_seconds: float = 300, deadline_seconds: float = 120,
                 cooldown_seconds: float = 60, max_pages: int = 50,
                 max_rows: int = 25000, min_rows: int = 500,
                 session_factory: Callable = requests.Session,
                 clock: Callable = time.monotonic):
        """限定缓存、源请求预算和完整性，支持离线注入。"""
        self.ttl_seconds, self.deadline_seconds = ttl_seconds, deadline_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_pages, self.max_rows, self.min_rows = max_pages, max_rows, min_rows
        self.session_factory, self.clock = session_factory, clock
        self.condition = threading.Condition()
        self.active = False
        self.cached: dict[str, Any] | None = None
        self.cached_at = 0.0
        self.failure_until = 0.0
        self.failure_message = ""
        self.fetch_count = self.request_count = self.failure_count = 0

    def status(self) -> dict[str, Any]:
        """仅读取来源状态，探活不发起上游请求。"""
        with self.condition:
            age = max(0.0, self.clock() - self.cached_at) if self.cached else None
            return {"active": self.active, "fetch_count": self.fetch_count,
                    "request_count": self.request_count, "failure_count": self.failure_count,
                    "cache_fresh": age is not None and age < self.ttl_seconds,
                    "cache_age_seconds": age, "cached_rows": len(self.cached["data"]) if self.cached else 0,
                    "fetched_at": self.cached["source_metadata"]["fetched_at"] if self.cached else None,
                    "cooldown_seconds": max(0.0, self.failure_until - self.clock()),
                    "deadline_seconds": self.deadline_seconds}

    def call(self) -> dict[str, Any]:
        """同一时刻只进行一次抓取，失败后冷却且不返回旧成功。"""
        wait_until = self.clock() + self.deadline_seconds + 5
        with self.condition:
            while self.active:
                remaining = wait_until - self.clock()
                if remaining <= 0:
                    raise RepurchaseSourceError("回购源正在运行，等待预算已耗尽")
                self.condition.wait(timeout=remaining)
            now = self.clock()
            if now < self.failure_until:
                raise RepurchaseSourceError(self.failure_message, self.failure_until - now)
            if self.cached and now - self.cached_at < self.ttl_seconds:
                return self._result(cache_hit=True)
            self.active = True
            self.fetch_count += 1
            previous_rows = len(self.cached["data"]) if self.cached else 0
        try:
            result = self._fetch(previous_rows)
        except Exception as exc:
            message = "回购源刷新失败: %s" % str(exc)
            with self.condition:
                self.failure_message = message
                self.failure_until = self.clock() + self.cooldown_seconds
                self.failure_count += 1
            raise RepurchaseSourceError(message, self.cooldown_seconds) from exc
        else:
            with self.condition:
                self.cached, self.cached_at = result, self.clock()
                self.failure_message, self.failure_until = "", 0.0
                return self._result(cache_hit=False)
        finally:
            with self.condition:
                self.active = False
                self.condition.notify_all()

    def _result(self, *, cache_hit: bool) -> dict[str, Any]:
        """复制缓存并保留真实取得时间和稳定内容身份。"""
        result = copy.deepcopy(self.cached)
        result["source_metadata"].update(cache_hit=cache_hit,
                                        cache_age_seconds=max(0.0, self.clock() - self.cached_at))
        return result

    def _fetch(self, previous_rows: int) -> dict[str, Any]:
        """按既有 AKShare 回购报表分页，不重拉用于发现页数的第一页。"""
        started = self.clock()
        deadline = started + self.deadline_seconds
        params = {"sortColumns": "UPD,DIM_DATE,DIM_SCODE", "sortTypes": "-1,-1,-1",
                  "pageSize": "500", "pageNumber": "1", "reportName": "RPTA_WEB_GETHGLIST_NEW",
                  "columns": "ALL", "source": "WEB"}
        count = 0
        with self.session_factory() as session:
            first, used = self._page(session, params, deadline)
            count += used
            pages = first.get("pages")
            if type(pages) is not int or not 1 <= pages <= self.max_pages:
                raise ValueError("回购源页数为空、无效或超过预算")
            expected = first.get("count")
            if expected is not None and (type(expected) is not int or not self.min_rows <= expected <= self.max_rows
                                         or math.ceil(expected / 500) != pages):
                raise ValueError("回购源总行数与页数不一致或超过预算")
            rows = []
            for page in range(1, pages + 1):
                if page == 1:
                    result = first
                else:
                    params["pageNumber"] = str(page)
                    result, used = self._page(session, params, deadline)
                    count += used
                if result.get("pages") != pages or result.get("count") != expected:
                    raise ValueError("回购源分页身份发生变化")
                data = result.get("data")
                if not isinstance(data, list) or not data or len(data) > 500 or (page < pages and len(data) != 500):
                    raise ValueError("回购源分页不完整")
                rows.extend(data)
                if len(rows) > self.max_rows:
                    raise ValueError("回购源行数超过预算")
        if len(rows) < self.min_rows or (expected is not None and len(rows) != expected):
            raise ValueError("回购源完整行数不符合元数据")
        if previous_rows and len(rows) < previous_rows * 0.8:
            raise ValueError("回购源行数较上次完整快照骤降，保留旧快照并阻止刷新")
        if self.clock() >= deadline:
            raise TimeoutError("回购源总预算耗尽")
        frame = self._frame(rows)
        if not all(isinstance(row, dict) and set(_COLUMN_MAP).issubset(row) for row in rows):
            raise ValueError("回购源字段合同不完整")
        if (not bool(frame["股票代码"].astype(str).str.fullmatch(r"\d{6}").all())
                or frame["最新公告日期"].isna().any()):
            raise ValueError("回购源代码或公告日期合同无效")
        split = json.loads(frame.to_json(orient="split", date_format="iso", force_ascii=False))
        result = {"operation": "stock_repurchase_em", "kind": "dataframe",
                  "columns": split["columns"], "data": split["data"]}
        snapshot = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        result["source_metadata"] = {"source": "eastmoney", "operation": "stock_repurchase_em",
                                     "fetched_at": datetime.now(timezone.utc).isoformat(),
                                     "snapshot_id": snapshot, "cache_hit": False, "cache_age_seconds": 0,
                                     "page_count": pages, "request_count": count,
                                     "elapsed_seconds": self.clock() - started,
                                     "deadline_seconds": self.deadline_seconds,
                                     "max_pages": self.max_pages, "max_rows": self.max_rows}
        result["source_metadata"].update(
            pages_complete=True, schema_verified=True, total_rows=len(rows),
            expected_total_rows=expected, page_size=500, legal_empty=False,
            query_scope={"operation": "stock_repurchase_em", "universe": "all_market",
                         "type": "full_snapshot", "report_name": params["reportName"]},
            schema_verification="akshare-1.18.81-repurchase-columns-key-date-v1")
        return result

    def _page(self, session, params: dict, deadline: float) -> tuple[dict, int]:
        """为每页限制连接、读等待和响应大小，只重试临时网络或服务异常。"""
        error = None
        for attempt in range(2):
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("回购源总预算耗尽")
            with self.condition:
                self.request_count += 1
            try:
                with session.get(_URL, params=dict(params), stream=True,
                                 timeout=(min(5, remaining / 2), min(15, remaining / 2))) as response:
                    response.raise_for_status()
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if self.clock() >= deadline:
                            raise TimeoutError("回购源总预算耗尽")
                        size += len(chunk)
                        if size > 2 * 1024 * 1024:
                            raise ValueError("回购源单页响应超过预算")
                        chunks.append(chunk)
                    payload = json.loads(b"".join(chunks))
                if not isinstance(payload, dict) or payload.get("success") is False or not isinstance(payload.get("result"), dict):
                    raise ValueError("回购源返回失败或缺少结果")
                return payload["result"], attempt + 1
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status != 429 and not 500 <= status <= 599:
                    raise
                error = exc
                if attempt == 0:
                    delay = min(1.0, max(0.0, deadline - self.clock()))
                    time.sleep(delay)
        raise error

    @staticmethod
    def _frame(rows: list[dict]) -> pd.DataFrame:
        """保持 AKShare 1.18.81 的列顺序、日期、数字和实施进度语义。"""
        frame = pd.DataFrame(rows).rename(columns=_COLUMN_MAP)[list(_COLUMN_MAP.values())]
        frame.insert(0, "序号", range(1, len(frame) + 1))
        frame["实施进度"] = frame["实施进度"].map(_PROGRESS)
        for field in ("回购起始时间", "最新公告日期"):
            frame[field] = pd.to_datetime(frame[field]).dt.date
        for field in _COLUMN_MAP.values():
            if field not in ("股票代码", "股票简称", "回购起始时间", "实施进度", "最新公告日期"):
                frame[field] = pd.to_numeric(frame[field])
        return frame


_SOURCE = RepurchaseSource()


def call_repurchase_source() -> dict[str, Any]:
    """返回同合同、带真实来源取得时间的完整回购快照。"""
    return _SOURCE.call()


def repurchase_source_status() -> dict[str, Any]:
    """为轻量健康接口投影活动、缓存和请求预算。"""
    return _SOURCE.status()
