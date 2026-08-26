from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app import create_app
from app.services.tdx_data_service import (
    artifact_path,
    call_baostock_history,
    start_market_snapshot,
)
from app.services.tdx_market_snapshot import publish_snapshot


class _AkshareFixture:
    """提供市场快照合同测试所需的固定 AKShare 响应。"""

    @staticmethod
    def stock_zh_a_spot_em():
        return pd.DataFrame({
            "涨跌幅": [1.0, -2.0, 0.0],
            "成交额": [100.0, 200.0, 300.0],
        })

    @staticmethod
    def stock_a_high_low_statistics():
        return pd.DataFrame({
            "日期": ["2026-07-31"],
            "20日新高": [5],
            "20日新低": [2],
        })

    @staticmethod
    def tool_trade_date_hist_sina():
        return pd.DataFrame({"trade_date": ["2026-07-31"]})

    @staticmethod
    def stock_zt_pool_em(date):
        return pd.DataFrame({"代码": ["000001"]})

    @staticmethod
    def stock_zt_pool_dtgc_em(date):
        return pd.DataFrame()

    @staticmethod
    def stock_zt_pool_zbgc_em(date):
        return pd.DataFrame()

    @staticmethod
    def stock_sector_fund_flow_rank(indicator, sector_type):
        return pd.DataFrame({
            "名称": ["银行"],
            "今日涨跌幅": [1.2],
            "今日主力净流入-净额": [100.0],
        })


class TdxDataApiTest(unittest.TestCase):
    """验证供 TDX 调用的机器 API 和结果包合同。"""

    def setUp(self):
        self.client = create_app().test_client()

    def test_health_supports_optional_bearer_auth(self):
        response = self.client.get("/api/tdx-data/health")
        self.assertEqual(response.status_code, 200)

        with patch.dict(
            os.environ, {"FINANCIAL_ANALYSIS_API_TOKEN": "secret"}, clear=False
        ):
            denied = self.client.get("/api/tdx-data/health")
            accepted = self.client.get(
                "/api/tdx-data/health",
                headers={"Authorization": "Bearer secret"},
            )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(accepted.status_code, 200)

    def test_create_job_accepts_only_market_snapshot(self):
        rejected = self.client.post(
            "/api/tdx-data/jobs", json={"job_type": "shareholders"})
        self.assertEqual(rejected.status_code, 400)

        with patch(
            "app.routes.tdx_data.start_market_snapshot",
            return_value={"job_id": "akshare-test", "running": True, "ok": None},
        ):
            accepted = self.client.post(
                "/api/tdx-data/jobs", json={"job_type": "market_snapshot"})

        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(accepted.get_json()["job_id"], "akshare-test")

    def test_start_job_uses_current_interpreter_and_shared_resource_lock(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"FINANCIAL_ANALYSIS_TDX_EXPORT_DIR": directory},
            clear=False,
        ), patch(
            "app.services.tdx_data_service.start_command_job",
            return_value=True,
        ) as start, patch(
            "app.services.tdx_data_service.get_job_state",
            return_value={"running": True, "ok": None},
        ):
            result = start_market_snapshot()

        self.assertTrue(result["running"])
        args, kwargs = start.call_args
        self.assertTrue(args[0].startswith("akshare-"))
        self.assertIn("app.services.tdx_market_snapshot", args[1])
        self.assertEqual(kwargs["resource_key"], "tdx-akshare-market-snapshot")

    def test_publish_snapshot_matches_tdx_manifest_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = publish_snapshot(
                directory,
                ak_module=_AkshareFixture(),
                snapshot_time=pd.Timestamp(
                    "2026-07-31 15:10:00", tz="Asia/Shanghai"),
                job_id="akshare-contract-test",
            )
            package = Path(directory) / "ready" / "akshare-contract-test"
            saved = json.loads(
                (package / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["protocol_version"], 1)
        self.assertEqual(saved["request"]["job_type"], "akshare_snapshot_export")
        self.assertEqual(
            {item["dataset"] for item in saved["files"]},
            {"akshare_market_state", "akshare_sector_flow"},
        )
        self.assertTrue(all(len(item["sha256"]) == 64 for item in saved["files"]))

    def test_artifact_path_rejects_unlisted_files(self):
        with self.assertRaisesRegex(ValueError, "不支持"):
            artifact_path("akshare-test", "../secret")

    def test_completed_artifact_remains_queryable_after_process_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"FINANCIAL_ANALYSIS_TDX_EXPORT_DIR": directory},
            clear=False,
        ):
            package = Path(directory) / "ready" / "akshare-restored"
            package.mkdir(parents=True)
            (package / "manifest.json").write_text(
                json.dumps({
                    "protocol_version": 1,
                    "job_id": "akshare-restored",
                    "status": "completed",
                    "files": [],
                }),
                encoding="utf-8",
            )
            with patch(
                "app.services.tdx_data_service.get_job_state",
                return_value={"running": False, "ok": None, "started_at": None},
            ):
                response = self.client.get(
                    "/api/tdx-data/jobs/akshare-restored")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

    def test_akshare_proxy_allows_only_named_dataframe_operations(self):
        fake_akshare = types.SimpleNamespace(
            fund_etf_hist_em=lambda **kwargs: pd.DataFrame({
                "日期": [kwargs["start_date"]],
                "成交额": [9000.0],
                "换手率": [1.25],
            }),
            stock_lhb_detail_em=lambda **kwargs: pd.DataFrame({
                "代码": ["000001"],
                "日期": [kwargs["start_date"]],
            }),
            stock_zh_a_hist_tx=lambda **_kwargs: pd.DataFrame({
                "date": ["2026-07-31"],
                "amount": [8000.0],
                "turnover": [0.0325],
            }),
            stock_zt_pool_em=lambda **kwargs: pd.DataFrame({
                "日期": [kwargs["date"]],
                "代码": ["000001"],
            }),
            stock_zt_pool_dtgc_em=lambda **kwargs: pd.DataFrame({
                "日期": [kwargs["date"]],
                "代码": ["000002"],
            }),
            stock_zt_pool_zbgc_em=lambda **kwargs: pd.DataFrame({
                "日期": [kwargs["date"]],
                "代码": ["000003"],
            }),
        )
        with patch.dict(sys.modules, {"akshare": fake_akshare}):
            response = self.client.post(
                "/api/tdx-data/akshare/stock_lhb_detail_em",
                json={"kwargs": {
                    "start_date": "20260731",
                    "end_date": "20260731",
                }},
            )
            tencent = self.client.post(
                "/api/tdx-data/akshare/stock_zh_a_hist_tx",
                json={"kwargs": {
                    "symbol": "sz000001",
                    "start_date": "20260731",
                    "end_date": "20260731",
                    "adjust": "",
                }},
            )
            etf = self.client.post(
                "/api/tdx-data/akshare/fund_etf_hist_em",
                json={"kwargs": {
                    "symbol": "510300",
                    "period": "daily",
                    "start_date": "20260731",
                    "end_date": "20260731",
                    "adjust": "",
                }},
            )
            limit_up = self.client.post(
                "/api/tdx-data/akshare/stock_zt_pool_em",
                json={"kwargs": {"date": "20260731"}},
            )
            limit_down = self.client.post(
                "/api/tdx-data/akshare/stock_zt_pool_dtgc_em",
                json={"kwargs": {"date": "20260731"}},
            )
            broken_limit = self.client.post(
                "/api/tdx-data/akshare/stock_zt_pool_zbgc_em",
                json={"kwargs": {"date": "20260731"}},
            )
        rejected = self.client.post(
            "/api/tdx-data/akshare/arbitrary_function",
            json={"kwargs": {}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["columns"], ["代码", "日期"])
        self.assertEqual(response.get_json()["data"][0][0], "000001")
        self.assertEqual(tencent.status_code, 200)
        self.assertEqual(
            tencent.get_json()["columns"], ["date", "amount", "turnover"])
        self.assertEqual(etf.status_code, 200)
        self.assertEqual(etf.get_json()["operation"], "fund_etf_hist_em")
        self.assertEqual(limit_up.status_code, 200)
        self.assertEqual(limit_up.get_json()["data"][0][1], "000001")
        self.assertEqual(limit_down.status_code, 200)
        self.assertEqual(limit_down.get_json()["data"][0][1], "000002")
        self.assertEqual(broken_limit.status_code, 200)
        self.assertEqual(broken_limit.get_json()["data"][0][1], "000003")
        self.assertEqual(rejected.status_code, 400)

    @patch("app.routes.tdx_data.call_baostock_history")
    def test_baostock_history_validates_and_returns_dataframe(self, history):
        """BaoStock 机器接口应透传规范历史表并拒绝非法代码。"""
        history.return_value = {
            "operation": "query_history_k_data_plus",
            "kind": "dataframe",
            "columns": ["date", "code", "amount", "turn"],
            "data": [["2026-07-31", "sz.000001", "8000", "3.25"]],
        }

        response = self.client.post(
            "/api/tdx-data/baostock/history",
            json={
                "symbol": "sz.000001",
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"][0][3], "3.25")
        history.assert_called_once_with(
            "sz.000001", "2026-07-01", "2026-07-31")

    def test_baostock_service_uses_unadjusted_daily_fields(self):
        """BaoStock 适配器应登录一次并返回不复权日线参考字段。"""
        class _Result:
            """模拟 BaoStock 游标结果。"""

            error_code = "0"
            error_msg = "success"
            fields = ["date", "code", "volume", "amount", "turn", "tradestatus"]

            def __init__(self):
                self.rows = [[
                    "2026-07-31", "sz.000001", "1000",
                    "8000", "3.25", "1",
                ]]

            def next(self):
                """按行推进模拟游标。"""
                return bool(self.rows)

            def get_row_data(self):
                """返回并移除当前模拟行。"""
                return self.rows.pop(0)

        fake_baostock = types.SimpleNamespace(
            login=lambda: types.SimpleNamespace(
                error_code="0", error_msg="success"),
            query_history_k_data_plus=lambda *_args, **_kwargs: _Result(),
            logout=lambda: None,
        )

        with patch.dict(sys.modules, {"baostock": fake_baostock}):
            result = call_baostock_history(
                "sz.000001", "2026-07-01", "2026-07-31")

        self.assertEqual(result["columns"], _Result.fields)
        self.assertEqual(result["data"][0][4], "3.25")
        with self.assertRaisesRegex(ValueError, "股票代码"):
            call_baostock_history(
                "../000001", "2026-07-01", "2026-07-31")


if __name__ == "__main__":
    unittest.main()
