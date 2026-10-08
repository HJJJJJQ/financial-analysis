from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import re
import textwrap
import threading
import time
import types
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import requests


_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORTS = {
    "stock_lhb_detail_em": "RPT_DAILYBILLBOARD_DETAILSNEW",
    "stock_lhb_hyyyb_em": "RPT_OPERATEDEPT_ACTIVE",
    "stock_lhb_jgmmtj_em": "RPT_ORGANIZATION_TRADE_DETAILS",
    "stock_lhb_stock_statistic_em": "RPT_BILLBOARD_TRADEALL",
    "stock_lhb_yybph_em": "RPT_RATEDEPT_RETURNT_RANKING",
}
_DATE_FIELDS = {"stock_lhb_detail_em": "上榜日", "stock_lhb_hyyyb_em": "上榜日",
                "stock_lhb_jgmmtj_em": "上榜日期", "stock_lhb_stock_statistic_em": "最近上榜日"}
_DATE_RAW_FIELDS = {"stock_lhb_detail_em": "TRADE_DATE", "stock_lhb_hyyyb_em": "ONLIST_DATE",
                   "stock_lhb_jgmmtj_em": "TRADE_DATE"}
_CYCLES = {"近一月": "01", "近三月": "02", "近六月": "03", "近一年": "04"}


class LhbSourceError(RuntimeError):
    """携带来源失败和有限退避，不将未证明的分页伪装为完整。"""

    def __init__(self, message: str, retry_after_seconds: float = 60):
        """保留机器接口可安全公开的来源错误。"""
        self.retry_after_seconds = max(1, math.ceil(retry_after_seconds))
        super().__init__(message)


class _VerifiedEmpty(Exception):
    """标记上游成功明确证明的空集合，绕开上游空表转换缺陷。"""


class _JsonResponse:
    """向未修改的 AKShare 转换函数提供已完整检查的内存页。"""

    def __init__(self, payload: dict):
        """保存不可被函数改坏的独立页响应。"""
        self.payload = payload

    def json(self):
        """按 requests 响应合同返回内存 JSON。"""
        return copy.deepcopy(self.payload)


def query_scope(operation: str, kwargs: dict) -> dict:
    """明确日期或滚动统计范围，不借函数过时默认日期当目标范围。"""
    scope = {"operation": operation, "universe": "all_market"}
    if operation in _DATE_RAW_FIELDS:
        if set(kwargs) != {"start_date", "end_date"}:
            raise ValueError("龙虎榜日期接口必须明确 start_date/end_date")
        for key in ("start_date", "end_date"):
            value = kwargs[key]
            if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
                raise ValueError("龙虎榜日期必须为 YYYYMMDD")
            scope[key] = datetime.strptime(value, "%Y%m%d").date().isoformat()
        if scope["start_date"] > scope["end_date"]:
            raise ValueError("龙虎榜开始日期晚于结束日期")
    else:
        if set(kwargs) - {"symbol"} or kwargs.get("symbol", "近一月") not in _CYCLES:
            raise ValueError("龙虎榜统计周期无效")
        scope.update(type="rolling_statistics", symbol=kwargs.get("symbol", "近一月"))
    return scope


def _output_columns(function: Callable) -> tuple[list[str], str]:
    """读取本地上游函数的最终固定列选择，保持列序与转换合同。"""
    source = textwrap.dedent(inspect.getsource(function))
    selections = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
            subscript = node.value.slice
            if isinstance(subscript, ast.List) and all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in subscript.elts):
                selections.append((node.lineno, [x.value for x in subscript.elts]))
    if not selections:
        raise LhbSourceError("龙虎榜 AKShare 输出列合同无法验证")
    return max(selections)[1], hashlib.sha256(source.encode()).hexdigest()


class _BoundedPages:
    """独立实例接管一轮请求，不修改全局 requests 或其他并发函数。"""

    def __init__(self, owner, session, operation, scope, deadline):
        """限定报告、范围、页数、行数与网络截止时间。"""
        self.owner, self.session, self.operation = owner, session, operation
        self.scope, self.deadline = scope, deadline
        self.pages = OrderedDict()
        self.base_params = None
        self.total_rows = self.request_count = 0
        self.raw_bytes = 0
        self.page_count = self.page_size = None
        self.raw_keys = None
        self.schema_digest = None

    def _scope_filter(self, params):
        """校验上游过滤表达式确实使用本次精确范围。"""
        actual = params.get("filter", "")
        if self.operation in _DATE_RAW_FIELDS:
            field = _DATE_RAW_FIELDS[self.operation]
            required = ("(%s>='%s')" % (field, self.scope["start_date"]),
                        "(%s<='%s')" % (field, self.scope["end_date"]))
            if len(actual) != sum(len(x) for x in required) or any(x not in actual for x in required):
                raise LhbSourceError("龙虎榜上游过滤范围不符合请求")
        else:
            field = "STATISTICS_CYCLE" if self.operation == "stock_lhb_stock_statistic_em" else "STATISTICSCYCLE"
            if actual != '(%s="%s")' % (field, _CYCLES[self.scope["symbol"]]):
                raise LhbSourceError("龙虎榜上游统计范围不符合请求")

    def get(self, url, *, params, **options):
        """首次调用完整取得分页，后续上游循环只读取内存同一页。"""
        if url != _URL or options or params.get("reportName") != REPORTS[self.operation]:
            raise LhbSourceError("龙虎榜函数请求了未批准的来源")
        self._scope_filter(params)
        stable = {key: value for key, value in params.items() if key != "pageNumber"}
        if self.base_params is not None and stable != self.base_params:
            raise LhbSourceError("龙虎榜函数改变了分页范围")
        if not self.pages:
            self.base_params = stable
            self._load(params)
        page = int(params.get("pageNumber", 1))
        if self.total_rows == 0:
            raise _VerifiedEmpty()
        if self.operation == "stock_lhb_stock_statistic_em":
            # 该上游函数仅读第一页；把已证明完整的集合交给其原列转换。
            payload = copy.deepcopy(self.pages[1])
            payload["result"]["data"] = [row for value in self.pages.values() for row in value["result"]["data"]]
            return _JsonResponse(payload)
        if page not in self.pages:
            raise LhbSourceError("龙虎榜函数读取了未证明的页")
        return _JsonResponse(self.pages[page])

    def _page(self, params):
        """显式限制连接、流式读取字节和总时间，临时网络至多重试一次。"""
        error = None
        for attempt in range(2):
            remaining = self.deadline - self.owner.clock()
            if remaining <= 0:
                raise LhbSourceError("龙虎榜来源总截止预算耗尽")
            self.request_count += 1
            try:
                with self.session.get(_URL, params=dict(params), stream=True,
                                      timeout=(min(5, remaining/2), min(15, remaining/2))) as response:
                    response.raise_for_status()
                    blocks, size = [], 0
                    for block in response.iter_content(chunk_size=8192):
                        size += len(block)
                        if size > self.owner.max_result_bytes or self.owner.clock() >= self.deadline:
                            raise LhbSourceError("龙虎榜单页字节或时间超过预算")
                        blocks.append(block)
                    payload = json.loads(b"".join(blocks))
                if not isinstance(payload, dict) or payload.get("success") is not True or not isinstance(payload.get("result"), dict):
                    raise LhbSourceError("龙虎榜上游未明确返回成功集合")
                return payload
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status != 429 and not 500 <= status <= 599:
                    raise LhbSourceError("龙虎榜上游拒绝请求") from exc
                error = exc
                if attempt == 0:
                    time.sleep(min(1, max(0, self.deadline-self.owner.clock())))
        raise LhbSourceError("龙虎榜上游临时网络错误") from error

    def _load(self, params):
        """逐页确认固定身份和精确行数，缺页、变动或截断均拒绝。"""
        first = self._page(dict(params, pageNumber="1"))
        result = first["result"]
        total, pages = result.get("count"), result.get("pages")
        size = int(params.get("pageSize", 0))
        if type(total) is not int or not 0 <= total <= self.owner.max_rows or not 1 <= size <= 5000:
            raise LhbSourceError("龙虎榜总行数或单页预算无效")
        if type(pages) is not int or not 0 <= pages <= self.owner.max_pages:
            raise LhbSourceError("龙虎榜页数无效或超过预算")
        if total == 0:
            if pages not in (0, 1) or result.get("data") != []:
                raise LhbSourceError("龙虎榜空集合缺少有效来源证明")
            self.pages[1] = first
            self.total_rows, self.page_count, self.page_size = 0, 1, size
            return
        first_data = result.get("data")
        if not isinstance(first_data, list) or not 1 <= len(first_data) <= size:
            raise LhbSourceError("龙虎榜首页缺失或超出单页预算")
        # 来源可能将 pageSize=5000 限制为 500；按真实首页推导固定页容量，仍须收齐 count 行。
        actual_size = len(first_data) if pages > 1 else size
        if pages != math.ceil(total/actual_size):
            raise LhbSourceError("龙虎榜总行数与分页不一致")
        collected = 0
        for page in range(1, pages+1):
            payload = first if page == 1 else self._page(dict(params, pageNumber=page))
            current = payload["result"]
            data = current.get("data")
            expected = min(actual_size, total-(page-1)*actual_size)
            if current.get("pages") != pages or current.get("count") != total or not isinstance(data, list) or len(data) != expected:
                raise LhbSourceError("龙虎榜分页缺失或集合发生变化")
            for row in data:
                if not isinstance(row, dict) or not row:
                    raise LhbSourceError("龙虎榜原始行字段无效")
                keys = list(row)
                if self.raw_keys is None:
                    self.raw_keys = keys
                if keys != self.raw_keys:
                    raise LhbSourceError("龙虎榜原始字段或字段顺序跨行变化")
                if self.operation in _DATE_RAW_FIELDS:
                    date = str(row.get(_DATE_RAW_FIELDS[self.operation], ""))[:10]
                    if not self.scope["start_date"] <= date <= self.scope["end_date"]:
                        raise LhbSourceError("龙虎榜原始日期超出本次范围")
            collected += len(data)
            self.raw_bytes += len(json.dumps(payload, ensure_ascii=False).encode())
            if self.raw_bytes > self.owner.max_result_bytes:
                raise LhbSourceError("龙虎榜原始集合超过字节预算")
            self.pages[page] = payload
        if collected != total:
            raise LhbSourceError("龙虎榜最终行数未达到来源总行数")
        self.total_rows, self.page_count, self.page_size = total, pages, actual_size
        self.schema_digest = hashlib.sha256(json.dumps(self.raw_keys).encode()).hexdigest()


class LhbSource:
    """五种龙虎榜来源共用有界所有权、同键合并与短期内存缓存。"""

    def __init__(self, *, deadline_seconds=120, ttl_seconds=60, cooldown_seconds=60,
                 max_pages=50, max_rows=25000, max_result_bytes=8*1024*1024,
                 max_cache_bytes=32*1024*1024, max_waiters=8,
                 session_factory=requests.Session, clock=time.monotonic):
        """注入网络和时钟，测试不会连接真实来源或数据库。"""
        self.deadline_seconds, self.ttl_seconds, self.cooldown_seconds = deadline_seconds, ttl_seconds, cooldown_seconds
        self.max_pages, self.max_rows, self.max_result_bytes = max_pages, max_rows, max_result_bytes
        self.max_cache_bytes, self.max_waiters = max_cache_bytes, max_waiters
        self.session_factory, self.clock = session_factory, clock
        self.condition = threading.Condition()
        self.active = False
        self.waiters = 0
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.failures = {}
        self.fetch_count = self.failure_count = 0

    def status(self):
        """探活只投影进程内预算，不触发网络或公开查询内容。"""
        with self.condition:
            return {"active": self.active, "waiters": self.waiters, "cache_entries": len(self.cache),
                    "cache_bytes": self.cache_bytes, "fetch_count": self.fetch_count,
                    "failure_count": self.failure_count, "deadline_seconds": self.deadline_seconds}

    def call(self, operation: str, kwargs: dict, function: Callable):
        """精确范围下串行取源，缓存命中保留首次 fetched_at 和身份。"""
        scope = query_scope(operation, kwargs)
        key = json.dumps(scope, sort_keys=True)
        deadline = self.clock()+self.deadline_seconds
        with self.condition:
            if self.waiters >= self.max_waiters:
                raise LhbSourceError("龙虎榜来源等待配额已满", 1)
            self.waiters += 1
            try:
                while self.active:
                    remaining = deadline-self.clock()
                    if remaining <= 0:
                        raise LhbSourceError("龙虎榜来源等待截止预算耗尽", 1)
                    self.condition.wait(timeout=remaining)
                now = self.clock()
                for expired, value in list(self.cache.items()):
                    if value[0]+self.ttl_seconds <= now:
                        self.cache_bytes -= value[2]
                        self.cache.pop(expired)
                for expired, value in list(self.failures.items()):
                    if value[0] <= now:
                        self.failures.pop(expired)
                if key in self.failures:
                    raise LhbSourceError("龙虎榜来源失败后处于冷却", self.failures[key][0]-now)
                if key in self.cache:
                    cached = self.cache[key]
                    value = copy.deepcopy(cached[1])
                    value["source_metadata"].update(cache_hit=True, cache_age_seconds=max(0, now-cached[0]))
                    self.cache.move_to_end(key)
                    return value
                self.active = True
                self.fetch_count += 1
            finally:
                self.waiters -= 1
        try:
            value = self._fetch(operation, kwargs, function, scope, deadline)
            size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
            if size > self.max_result_bytes:
                raise LhbSourceError("龙虎榜转换结果超过字节预算")
            with self.condition:
                while self.cache and (self.cache_bytes+size > self.max_cache_bytes or len(self.cache) >= 16):
                    _, previous = self.cache.popitem(last=False)
                    self.cache_bytes -= previous[2]
                if size <= self.max_cache_bytes:
                    self.cache[key] = (self.clock(), copy.deepcopy(value), size)
                    self.cache_bytes += size
                return value
        except Exception as exc:
            with self.condition:
                if len(self.failures) >= 32:
                    self.failures.pop(next(iter(self.failures)))
                self.failures[key] = (self.clock()+self.cooldown_seconds,)
                self.failure_count += 1
            if isinstance(exc, LhbSourceError):
                raise
            raise LhbSourceError("龙虎榜来源字段或转换合同失效") from exc
        finally:
            with self.condition:
                self.active = False
                self.condition.notify_all()

    def _fetch(self, operation, kwargs, function, scope, deadline):
        """实例克隆函数只替换网络依赖，原始 AKShare 列和值转换保持。"""
        columns, transform_sha256 = _output_columns(function)
        with self.session_factory() as session:
            adapter = _BoundedPages(self, session, operation, scope, deadline)
            namespace = dict(function.__globals__, requests=adapter,
                             get_tqdm=lambda: lambda iterable, **_options: iterable)
            copied = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
            copied.__kwdefaults__ = function.__kwdefaults__
            try:
                frame = copied(**kwargs)
            except _VerifiedEmpty:
                frame = pd.DataFrame(columns=columns)
            if self.clock() >= deadline or adapter.page_count is None:
                raise LhbSourceError("龙虎榜来源未完成有效分页读取")
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns or len(frame) != adapter.total_rows:
            raise LhbSourceError("龙虎榜最终字段或行数合同不一致")
        if len(frame):
            if "代码" in columns and not bool(frame["代码"].astype(str).str.fullmatch(r"\d{6}").all()):
                raise LhbSourceError("龙虎榜股票代码转换无效")
            if operation in _DATE_FIELDS and frame[_DATE_FIELDS[operation]].isna().any():
                raise LhbSourceError("龙虎榜日期转换缺失")
        split = json.loads(frame.to_json(orient="split", date_format="iso", force_ascii=False))
        value = {"operation": operation, "kind": "dataframe", "columns": split["columns"], "data": split["data"]}
        snapshot = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        value["source_metadata"] = {"source": "eastmoney", "operation": operation,
            "fetched_at": datetime.now(timezone.utc).isoformat(), "snapshot_id": snapshot,
            "cache_hit": False, "cache_age_seconds": 0, "pages_complete": True, "schema_verified": True,
            "schema_verification": "akshare-output-columns-and-key-date-v1", "transform_sha256": transform_sha256,
            "source_schema_sha256": adapter.schema_digest, "query_scope": scope,
            "report_name": REPORTS[operation], "page_count": adapter.page_count,
            "page_size": adapter.page_size, "total_rows": adapter.total_rows, "expected_total_rows": adapter.total_rows,
            "request_count": adapter.request_count, "deadline_seconds": self.deadline_seconds,
            "max_pages": self.max_pages, "max_rows": self.max_rows,
            "legal_empty": adapter.total_rows == 0,
            "verified_empty_reason": "upstream_success_count_zero_data_empty" if adapter.total_rows == 0 else None}
        return value


_SOURCE = LhbSource()


def call_lhb_source(operation: str, kwargs: dict, function: Callable):
    """只为五种允许的真实 AKShare 来源提供完整分页证据。"""
    if (not isinstance(function, types.FunctionType) or "requests" not in function.__globals__
            or getattr(function, "__module__", "") != "akshare.stock_feature.stock_lhb_em"):
        # 离线注入函数或未知实现保持原兼容返回，但明确不构造来源证明。
        frame = function(**kwargs)
        frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        split = json.loads(frame.to_json(orient="split", date_format="iso", force_ascii=False))
        return {"operation": operation, "kind": "dataframe", "columns": split["columns"], "data": split["data"],
                "source_metadata": {"operation": operation, "pages_complete": False, "schema_verified": False,
                                    "proof_missing_reason": "source_transport_not_observed"}}
    return _SOURCE.call(operation, kwargs, function)


def lhb_source_status():
    """投影来源状态，不触发上游请求。"""
    return _SOURCE.status()
