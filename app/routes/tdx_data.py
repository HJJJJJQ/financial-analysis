from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file

from app.services.tdx_data_service import (
    artifact_path,
    authorize_machine_request,
    call_akshare_operation,
    call_baostock_history,
    machine_job_state,
    start_market_snapshot,
)
from app.services.repurchase_source import RepurchaseSourceError, repurchase_source_status


bp = Blueprint("tdx_data", __name__)


@bp.before_request
def require_machine_auth():
    """为机器接口应用可选 Bearer Token 认证。"""
    if not authorize_machine_request(request):
        abort(401)


@bp.get("/api/tdx-data/health")
def health():
    """返回独立采集后端的轻量健康状态。"""
    return jsonify({"status": "ok", "service": "financial-analysis",
                    "repurchase_source": repurchase_source_status()})


@bp.post("/api/tdx-data/jobs")
def create_job():
    """提交 TDX 数据采集任务。"""
    payload = request.get_json(silent=True) or {}
    if payload.get("job_type") != "market_snapshot":
        return jsonify({"error": "仅支持 market_snapshot"}), 400
    try:
        return jsonify(start_market_snapshot()), 202
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409


@bp.get("/api/tdx-data/jobs/<job_id>")
def job_status(job_id: str):
    """查询 TDX 数据采集任务状态。"""
    try:
        return jsonify(machine_job_state(job_id))
    except FileNotFoundError:
        abort(404)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/api/tdx-data/jobs/<job_id>/manifest")
def download_manifest(job_id: str):
    """下载任务 manifest。"""
    return _send_artifact(job_id, "manifest.json")


@bp.get("/api/tdx-data/jobs/<job_id>/files/<filename>")
def download_file(job_id: str, filename: str):
    """下载 manifest 明确列出的标准结果文件。"""
    return _send_artifact(job_id, filename)


@bp.post("/api/tdx-data/akshare/<operation>")
def call_akshare(operation: str):
    """执行供 TDX 旧落库流程使用的白名单 AKShare 表接口。"""
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(call_akshare_operation(
            operation, payload.get("kwargs") or {}))
    except RepurchaseSourceError as exc:
        return jsonify({"error": str(exc), "code": "repurchase_source_unavailable"}), 502, {
            "Retry-After": str(exc.retry_after_seconds)}
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/api/tdx-data/baostock/history")
def call_baostock():
    """执行供 TDX 缺口回补使用的 BaoStock 历史行情接口。"""
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(call_baostock_history(
            payload.get("symbol"),
            payload.get("start_date"),
            payload.get("end_date"),
        ))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


def _send_artifact(job_id: str, filename: str):
    """校验并发送单个任务产物。"""
    try:
        path = artifact_path(job_id, filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name, max_age=0)
