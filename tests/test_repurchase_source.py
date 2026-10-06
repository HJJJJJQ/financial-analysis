from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import requests
from flask import Flask

from app.routes.tdx_data import bp
from app.services.repurchase_source import RepurchaseSource, RepurchaseSourceError


def _row(index=1):
    """构造包含已验证回购报表字段的单条原始记录。"""
    return {"DIM_SCODE": "%06d" % index, "SECURITYSHORTNAME": "样本",
            "NEWPRICE": "10.5", "REPURPRICECAP": "12", "REPURNUMLOWER": "1000",
            "REPURNUMCAP": "2000", "ZSZXX": "0.1", "ZSZSX": "0.2", "JEXX": "10000",
            "JESX": "20000", "DIM_TRADEDATE": "2026-09-01", "REPURPROGRESS": "004",
            "REPURPRICELOWER1": "9", "REPURPRICECAP1": "11", "REPURNUM": "1500",
            "REPURAMOUNT": "15000", "UPDATEDATE": "2026-09-30"}


class _Response:
    """模拟流式响应，测试不联网。"""

    def __init__(self, payload):
        """保存固定 JSON 页。"""
        self.payload = payload

    def __enter__(self):
        """返回上下文响应。"""
        return self

    def __exit__(self, *args):
        """结束离线响应。"""
        return

    def raise_for_status(self):
        """固定成功页无需转换 HTTP 错误。"""
        return

    def iter_content(self, chunk_size):
        """返回固定字节片段。"""
        yield json.dumps(self.payload).encode()


class _Session:
    """记录分页请求并支持故障和并发等待。"""

    def __init__(self, rows, *, entered=None, release=None):
        """保存完整快照和可选并发屏障。"""
        self.rows, self.entered, self.release = rows, entered, release
        self.calls = []
        self.error = None
        self.page_override = {}

    def __enter__(self):
        """返回离线会话。"""
        return self

    def __exit__(self, *args):
        """关闭离线会话。"""
        return

    def get(self, url, *, params, timeout, stream):
        """按页返回固定数据，记录显式超时参数。"""
        page = int(params["pageNumber"])
        self.calls.append({"page": page, "timeout": timeout, "stream": stream})
        if self.entered:
            self.entered.set()
            if not self.release.wait(timeout=3):
                raise TimeoutError("测试屏障未释放")
        if self.error:
            raise self.error
        pages = (len(self.rows) + 499) // 500
        result = {"pages": pages, "count": len(self.rows),
                  "data": self.rows[(page - 1) * 500:page * 500]}
        result.update(self.page_override.get(page, {}))
        return _Response({"success": True, "result": result})


class RepurchaseSourceTests(unittest.TestCase):
    """核对回购合同、完整性、同源合并、缓存和有界失败。"""

    def test_preserves_contract_and_fetches_first_page_once(self):
        """保留原列和值语义，发现页数的第一页不能重复请求。"""
        session = _Session([_row(i) for i in range(1, 502)])
        source = RepurchaseSource(session_factory=lambda: session, min_rows=1)
        value = source.call()
        self.assertEqual([x["page"] for x in session.calls], [1, 2])
        self.assertTrue(all(x["timeout"][0] <= 5 and x["timeout"][1] <= 15 and x["stream"]
                            for x in session.calls))
        self.assertEqual(value["columns"], ["序号", "股票代码", "股票简称", "最新价", "计划回购价格区间",
            "计划回购数量区间-下限", "计划回购数量区间-上限", "占公告前一日总股本比例-下限",
            "占公告前一日总股本比例-上限", "计划回购金额区间-下限", "计划回购金额区间-上限",
            "回购起始时间", "实施进度", "已回购股份价格区间-下限", "已回购股份价格区间-上限",
            "已回购股份数量", "已回购金额", "最新公告日期"])
        self.assertEqual(value["data"][0], [1, "000001", "样本", 10.5, 12, 1000, 2000, 0.1, 0.2,
            10000, 20000, "2026-09-01T00:00:00.000", "实施中", 9, 11, 1500, 15000,
            "2026-09-30T00:00:00.000"])
        self.assertEqual(value["source_metadata"]["page_count"], 2)
        self.assertEqual(value["source_metadata"]["request_count"], 2)

    def test_cache_preserves_actual_fetch_time_and_isolated_payload(self):
        """缓存命中保留同一观察身份，调用方不能改坏内部缓存。"""
        session = _Session([_row()])
        source = RepurchaseSource(session_factory=lambda: session, min_rows=1)
        first = source.call()
        identity = dict(first["source_metadata"])
        first["data"][0][1] = "bad"
        second = source.call()
        self.assertTrue(second["source_metadata"]["cache_hit"])
        self.assertEqual(second["source_metadata"]["fetched_at"], identity["fetched_at"])
        self.assertEqual(second["source_metadata"]["snapshot_id"], identity["snapshot_id"])
        self.assertEqual(second["data"][0][1], "000001")
        self.assertEqual(len(session.calls), 1)

    def test_concurrent_requests_share_one_source_fetch(self):
        """三个同时请求只抓取一轮，等待者复用原来源观察。"""
        entered, release = threading.Event(), threading.Event()
        session = _Session([_row()], entered=entered, release=release)
        source = RepurchaseSource(session_factory=lambda: session, min_rows=1)
        with ThreadPoolExecutor(max_workers=3) as pool:
            leader = pool.submit(source.call)
            self.assertTrue(entered.wait(timeout=2))
            waiting = [pool.submit(source.call) for _ in range(2)]
            self.assertTrue(source.status()["active"])
            release.set()
            values = [leader.result(timeout=3)] + [x.result(timeout=3) for x in waiting]
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len({x["source_metadata"]["snapshot_id"] for x in values}), 1)
        self.assertEqual(sum(not x["source_metadata"]["cache_hit"] for x in values), 1)

    def test_deadline_expires_without_upstream_request(self):
        """来源预算已耗尽时不允许补发请求或生成成功缓存。"""
        session = _Session([_row()])
        source = RepurchaseSource(session_factory=lambda: session, min_rows=1, deadline_seconds=0)
        with self.assertRaises(RepurchaseSourceError):
            source.call()
        self.assertEqual(session.calls, [])
        self.assertFalse(source.status()["cache_fresh"])

    def test_timeout_retries_are_bounded_and_cooldown_is_visible(self):
        """单页失败至多尝试两次，后续请求处于冷却且不能返回成功。"""
        session = _Session([_row()])
        session.error = requests.Timeout("固定网络超时")
        source = RepurchaseSource(session_factory=lambda: session, min_rows=1)
        with patch("app.services.repurchase_source.time.sleep"):
            with self.assertRaises(RepurchaseSourceError):
                source.call()
        with self.assertRaises(RepurchaseSourceError) as error:
            source.call()
        self.assertEqual(len(session.calls), 2)
        self.assertGreater(error.exception.retry_after_seconds, 0)
        self.assertEqual(source.status()["failure_count"], 1)

    def test_incomplete_changed_and_excessive_pages_do_not_cache_success(self):
        """分页缺失、集合变动或超限全部拒绝成功。"""
        for override in ({"data": []}, {"count": 502}, {"pages": 51}):
            with self.subTest(override=override):
                session = _Session([_row(i) for i in range(1, 502)])
                session.page_override[1 if "pages" in override else 2] = override
                source = RepurchaseSource(session_factory=lambda: session, min_rows=1)
                with self.assertRaises(RepurchaseSourceError):
                    source.call()
                self.assertFalse(source.status()["cache_fresh"])

    def test_snapshot_drop_preserves_old_identity_without_serving_it_as_success(self):
        """完整快照骤降时保留旧证据，刷新响应仍明确失败。"""
        now = [10.0]
        session = _Session([_row(i) for i in range(1, 502)])
        source = RepurchaseSource(session_factory=lambda: session, min_rows=1,
                                  ttl_seconds=5, clock=lambda: now[0])
        initial = source.call()["source_metadata"]
        now[0] += 6
        session.rows = [_row()]
        with self.assertRaises(RepurchaseSourceError):
            source.call()
        self.assertEqual(source.status()["fetched_at"], initial["fetched_at"])
        self.assertFalse(source.status()["cache_fresh"])

    def test_http_health_and_error_do_not_fetch_source(self):
        """轻量探活不抓源，来源失败返回502及有限退避提示。"""
        app = Flask(__name__)
        app.register_blueprint(bp)
        with patch("app.services.repurchase_source.call_repurchase_source",
                   side_effect=RepurchaseSourceError("上游预算耗尽", 60)) as fetch:
            with app.test_client() as client:
                health = client.get("/api/tdx-data/health")
                self.assertEqual(health.status_code, 200)
                fetch.assert_not_called()
                response = client.post("/api/tdx-data/akshare/stock_repurchase_em", json={"kwargs": {}})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.headers["Retry-After"], "60")
        self.assertEqual(response.get_json()["code"], "repurchase_source_unavailable")


if __name__ == "__main__":
    unittest.main()
