from __future__ import annotations

import ast
import copy
import inspect
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import akshare
import requests
from flask import Flask

from app.routes.tdx_data import bp
from app.services.lhb_source import LhbSource, LhbSourceError, REPORTS, _output_columns


def _raw_row(operation, index=1):
    """按本地 AKShare 固定转换创建原始行，日期与股票代码具有真实形状。"""
    function = getattr(akshare, operation)
    tree = ast.parse(inspect.getsource(function))
    ordered = None
    mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            if any(isinstance(target, ast.Attribute) and target.attr == "columns" for target in node.targets):
                ordered = [value.value for value in node.value.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rename":
            for keyword in node.keywords:
                if keyword.arg == "columns" and isinstance(keyword.value, ast.Dict):
                    mapping = ast.literal_eval(keyword.value)
    if ordered:
        row = {"RAW_%02d" % position: "1" for position in range(len(ordered)-1)}
        keys = list(row)
        for column, field, value in (("代码", "SECURITY_CODE", "%06d" % index),
                ("上榜日", "ONLIST_DATE", "2026-09-30"), ("上榜日期", "TRADE_DATE", "2026-09-30"),
                ("最近上榜日", "LATEST_TDATE", "2026-09-30")):
            if column in ordered:
                keys[ordered.index(column)-1] = field
        row = {key: "1" for key in keys}
        for column, field, value in (("代码", "SECURITY_CODE", "%06d" % index),
                ("上榜日", "ONLIST_DATE", "2026-09-30"), ("上榜日期", "TRADE_DATE", "2026-09-30"),
                ("最近上榜日", "LATEST_TDATE", "2026-09-30")):
            if field in row:
                row[field] = value
        return row
    row = {key: "1" for key in mapping if key != "index"}
    if operation == "stock_lhb_detail_em":
        row.update(SECURITY_CODE="%06d" % index, SECURITY_NAME_ABBR="样本", TRADE_DATE="2026-09-30")
    else:
        row["OPERATEDEPT_NAME"] = "样本营业部"
    return row


class _Response:
    """仅返回 fixture 流式 JSON，不访问网络。"""

    def __init__(self, value):
        """保存固定响应。"""
        self.value = value

    def __enter__(self):
        """提供 requests 流上下文。"""
        return self

    def __exit__(self, *args):
        """离线结束响应。"""
        return None

    def raise_for_status(self):
        """fixture 不模拟其他 HTTP 状态。"""
        return None

    def iter_content(self, chunk_size):
        """输出完整固定字节。"""
        yield json.dumps(self.value).encode()


class _Session:
    """模拟按固定报告取页，并记录超时、请求次数和并发屏障。"""

    def __init__(self, rows, *, entered=None, release=None):
        """准备可变分页 fixture 和可选同键并发等待。"""
        self.rows, self.entered, self.release = rows, entered, release
        self.calls, self.override = [], {}
        self.error = None

    def __enter__(self):
        """提供 fixture 会话。"""
        return self

    def __exit__(self, *args):
        """关闭会话。"""
        return None

    def get(self, url, *, params, stream, timeout):
        """每页只取得一次，不接受不受限请求。"""
        page, size = int(params["pageNumber"]), int(params["pageSize"])
        self.calls.append((page, size, stream, timeout))
        if self.entered:
            self.entered.set()
            if not self.release.wait(3):
                raise RuntimeError("fixture 并发屏障耗尽")
        if self.error:
            raise self.error
        value = {"success": True, "result": {"pages": (len(self.rows)+size-1)//size,
                 "count": len(self.rows), "data": self.rows[(page-1)*size:page*size]}}
        value["result"].update(self.override.get(page, {}))
        return _Response(value)


class LhbSourceTests(unittest.TestCase):
    """检验真实转换兼容、分页证明、合法空、预算与实例隔离。"""

    @staticmethod
    def _kwargs(operation):
        """日期族必须使用显式范围，统计族使用原周期默认值。"""
        return {"symbol": "近一月"} if operation in ("stock_lhb_stock_statistic_em", "stock_lhb_yybph_em") else {"start_date": "20260930", "end_date": "20260930"}

    def test_all_five_original_transforms_and_scopes_are_preserved(self):
        """五族复用上游列转换，来源 metadata 只增加证据。"""
        for operation in REPORTS:
            with self.subTest(operation=operation):
                function = getattr(akshare, operation)
                original_transport = function.__globals__["requests"]
                session = _Session([_raw_row(operation)])
                value = LhbSource(session_factory=lambda: session).call(operation, self._kwargs(operation), function)
                self.assertEqual(value["columns"], _output_columns(function)[0])
                self.assertEqual(len(value["data"]), 1)
                metadata = value["source_metadata"]
                self.assertIs(metadata["pages_complete"], True)
                self.assertIs(metadata["schema_verified"], True)
                self.assertEqual(metadata["total_rows"], 1)
                self.assertEqual(metadata["operation"], operation)
                self.assertEqual(metadata["query_scope"]["universe"], "all_market")
                if "start_date" in self._kwargs(operation):
                    self.assertEqual(metadata["query_scope"]["start_date"], "2026-09-30")
                self.assertIs(function.__globals__["requests"], original_transport)
                self.assertEqual(len(session.calls), 1)
                self.assertTrue(session.calls[0][2])

    def test_statistic_reads_all_pages_instead_of_silent_first_page(self):
        """单页上游函数仍转换全部取得的源行，固定列序不变。"""
        operation = "stock_lhb_stock_statistic_em"
        session = _Session([_raw_row(operation, i) for i in range(1, 5002)])
        value = LhbSource(session_factory=lambda: session).call(operation, self._kwargs(operation), getattr(akshare, operation))
        self.assertEqual([call[0] for call in session.calls], [1, 2])
        self.assertEqual(len(value["data"]), 5001)
        self.assertEqual(value["data"][-1][value["columns"].index("代码")], "005001")
        self.assertEqual(value["source_metadata"]["page_count"], 2)

    def test_page_probe_is_reused_and_missing_page_rejected(self):
        """循环分页族不重复第一页，第二页空不能伪 complete。"""
        operation = "stock_lhb_jgmmtj_em"
        session = _Session([_raw_row(operation, i) for i in range(1, 502)])
        source = LhbSource(session_factory=lambda: session)
        value = source.call(operation, self._kwargs(operation), getattr(akshare, operation))
        self.assertEqual(len(value["data"]), 501)
        self.assertEqual([call[0] for call in session.calls], [1, 2])
        session = _Session([_raw_row(operation, i) for i in range(1, 502)])
        session.override[2] = {"data": []}
        source = LhbSource(session_factory=lambda: session)
        with self.assertRaises(LhbSourceError):
            source.call(operation, self._kwargs(operation), getattr(akshare, operation))
        self.assertEqual(source.status()["cache_entries"], 0)

    def test_explicit_success_count_zero_returns_valid_columns_and_proof(self):
        """空表保留各族固定字段，只有来源明确为空才标 legal_empty。"""
        for operation in REPORTS:
            with self.subTest(operation=operation):
                function = getattr(akshare, operation)
                value = LhbSource(session_factory=lambda: _Session([])).call(operation, self._kwargs(operation), function)
                self.assertEqual(value["columns"], _output_columns(function)[0])
                self.assertEqual(value["data"], [])
                self.assertIs(value["source_metadata"]["legal_empty"], True)
                self.assertEqual(value["source_metadata"]["verified_empty_reason"], "upstream_success_count_zero_data_empty")
        session = _Session([])
        session.override[1] = {"count": None}
        with self.assertRaises(LhbSourceError):
            LhbSource(session_factory=lambda: session).call("stock_lhb_detail_em", self._kwargs("stock_lhb_detail_em"), akshare.stock_lhb_detail_em)

    def test_singleflight_cache_retains_original_source_time_and_identity(self):
        """两并发调用共用源请求，缓存不刷新 PIT 来源时刻。"""
        operation = "stock_lhb_detail_em"
        entered, release = threading.Event(), threading.Event()
        session = _Session([_raw_row(operation)], entered=entered, release=release)
        source = LhbSource(session_factory=lambda: session)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(source.call, operation, self._kwargs(operation), getattr(akshare, operation))
            self.assertTrue(entered.wait(2))
            second = pool.submit(source.call, operation, self._kwargs(operation), getattr(akshare, operation))
            release.set()
            values = [first.result(3), second.result(3)]
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(values[0]["source_metadata"]["fetched_at"], values[1]["source_metadata"]["fetched_at"])
        self.assertEqual(values[0]["source_metadata"]["snapshot_id"], values[1]["source_metadata"]["snapshot_id"])
        self.assertEqual(sum(value["source_metadata"]["cache_hit"] for value in values), 1)
        values[0]["data"].clear()
        self.assertEqual(len(source.call(operation, self._kwargs(operation), getattr(akshare, operation))["data"]), 1)

    def test_page_count_schema_date_and_resource_limits_reject_success(self):
        """字段/日期漂移、分页变动、时间与容量超限不进入缓存。"""
        operation = "stock_lhb_detail_em"
        for override in ({"pages": 51}, {"pages": 2}, {"count": None}, {"count": 25001}):
            with self.subTest(override=override):
                session = _Session([_raw_row(operation)])
                session.override[1] = override
                with self.assertRaises(LhbSourceError):
                    LhbSource(session_factory=lambda: session).call(operation, self._kwargs(operation), getattr(akshare, operation))
        for update in ({"TRADE_DATE": "2026-10-01"}, {"SECURITY_CODE": "invalid"}):
            row = _raw_row(operation)
            row.update(update)
            with self.assertRaises(LhbSourceError):
                LhbSource(session_factory=lambda: _Session([row])).call(operation, self._kwargs(operation), getattr(akshare, operation))
        with self.assertRaises(LhbSourceError):
            LhbSource(session_factory=lambda: _Session([_raw_row(operation)]), max_result_bytes=1).call(operation, self._kwargs(operation), getattr(akshare, operation))
        session = _Session([_raw_row(operation)])
        with self.assertRaises(LhbSourceError):
            LhbSource(session_factory=lambda: session, deadline_seconds=0).call(operation, self._kwargs(operation), getattr(akshare, operation))
        self.assertEqual(session.calls, [])

    def test_temporary_retries_and_cooldown_are_bounded(self):
        """临时网络只重试一次，后续请求冷却而非运行无限循环。"""
        operation = "stock_lhb_detail_em"
        session = _Session([_raw_row(operation)])
        session.error = requests.Timeout("fixture")
        source = LhbSource(session_factory=lambda: session)
        with patch("app.services.lhb_source.time.sleep"):
            with self.assertRaises(LhbSourceError):
                source.call(operation, self._kwargs(operation), getattr(akshare, operation))
        with self.assertRaises(LhbSourceError):
            source.call(operation, self._kwargs(operation), getattr(akshare, operation))
        self.assertEqual(len(session.calls), 2)

    def test_health_does_not_fetch_and_http_source_error_has_retry_after(self):
        """健康只读取内存，失败有明确来源代码与退避。"""
        app = Flask(__name__)
        app.register_blueprint(bp)
        with patch("app.services.lhb_source.call_lhb_source", side_effect=LhbSourceError("fixture", 60)) as fetch:
            with app.test_client() as client:
                self.assertEqual(client.get("/api/tdx-data/health").status_code, 200)
                fetch.assert_not_called()
                response = client.post("/api/tdx-data/akshare/stock_lhb_detail_em", json={"kwargs": self._kwargs("stock_lhb_detail_em")})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.headers["Retry-After"], "60")
        self.assertEqual(response.get_json()["code"], "lhb_source_unavailable")


if __name__ == "__main__":
    unittest.main()
