#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate the DeepEP DCDB ``config.json`` from a performance summary CSV.

The performance rows remain in a separate CSV file.  The DeepEP DCDB schema is
embedded below, based on the approved ``yml/config.json`` shape; the referenced
file is not read at runtime.

The input must be the reviewed 36-column summary emitted by
``extract_deepep_summary.yml``.

Examples::

    python scripts/batch_convert.py \
        deepep_performance_2026-07-14.csv \
        --system-info deepep_perf_extract/system_info.json \
        --output-dir deepep_dcdb_upload \
        --dailydate 2026-07-14 \
        --log-path /data/swqa/swqa_shared_dir/deepep_daily/master/2026-07-14/DeepEP/pytest/test_log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NoReturn, Sequence


SUPPORTED_MODES = {"intranode_normal", "internode_normal", "low_latency"}
LATEST_COLUMNS = [
    "primitive",
    "implementation",
    "test_mode",
    "protocol",
    "rank_num",
    "tokens",
    "hidden_size",
    "top_k",
    "experts",
    "communication_scope",
    "network_link",
    "nic_handler",
    "noise_type",
    "routing_mode",
    "dispatch_dtype",
    "combine_dtype",
    "dispatch_e2e_bandwidth_gbps",
    "dispatch_e2e_total_latency_us",
    "dispatch_send_latency_us",
    "dispatch_recv_latency_us",
    "combine_e2e_bandwidth_gbps",
    "combine_e2e_total_latency_us",
    "combine_send_latency_us",
    "combine_recv_latency_us",
    "dispatch_sms",
    "dispatch_nvl_chunk",
    "dispatch_rdma_chunk",
    "dispatch_rdma_bandwidth_gbps",
    "dispatch_nvl_bandwidth_gbps",
    "dispatch_latency_us",
    "combine_sms",
    "combine_nvl_chunk",
    "combine_rdma_chunk",
    "combine_rdma_bandwidth_gbps",
    "combine_nvl_bandwidth_gbps",
    "combine_latency_us",
]
NUMERIC_COLUMNS = {
    "rank_num", "tokens", "hidden_size", "top_k", "experts",
    "dispatch_e2e_bandwidth_gbps", "dispatch_e2e_total_latency_us",
    "dispatch_send_latency_us", "dispatch_recv_latency_us",
    "combine_e2e_bandwidth_gbps", "combine_e2e_total_latency_us",
    "combine_send_latency_us", "combine_recv_latency_us", "dispatch_sms",
    "dispatch_nvl_chunk", "dispatch_rdma_chunk", "dispatch_rdma_bandwidth_gbps",
    "dispatch_nvl_bandwidth_gbps", "dispatch_latency_us", "combine_sms",
    "combine_nvl_chunk", "combine_rdma_chunk", "combine_rdma_bandwidth_gbps",
    "combine_nvl_bandwidth_gbps", "combine_latency_us",
}
UNIFIED_COLUMNS = {
    "dispatch_e2e_bandwidth_gbps", "dispatch_e2e_total_latency_us",
    "combine_e2e_bandwidth_gbps", "combine_e2e_total_latency_us",
}
SEND_RECV_COLUMNS = {
    "dispatch_send_latency_us", "dispatch_recv_latency_us",
    "combine_send_latency_us", "combine_recv_latency_us",
}
TUNING_COLUMNS = {
    "dispatch_sms", "dispatch_nvl_chunk", "dispatch_rdma_chunk",
    "dispatch_rdma_bandwidth_gbps", "dispatch_nvl_bandwidth_gbps",
    "dispatch_latency_us", "combine_sms", "combine_nvl_chunk",
    "combine_rdma_chunk", "combine_rdma_bandwidth_gbps",
    "combine_nvl_bandwidth_gbps", "combine_latency_us",
}
NUMERIC_RE = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


def param(name: str, direction: str, required: bool, data_type: str, unit: str, description: str) -> Dict[str, Any]:
    return {
        "id": None,
        "operator_id": None,
        "name": name,
        "param_direction": direction,
        "required": required,
        "data_type": data_type,
        "unit": unit,
        "default_value": None,
        "description": description,
    }


DEFAULT_OPERATOR_PARAM_DEF = [
    param("test_mode", "input", True, "string", "", "测试模式"),
    param("protocol", "input", True, "string", "", "通信协议"),
    param("rank_num", "input", True, "int", "", "进程数量"),
    param("tokens", "input", True, "int", "", "令牌数量"),
    param("hidden_size", "input", True, "int", "", "隐藏层维度"),
    param("top_k", "input", True, "int", "", "每个令牌选择的专家数"),
    param("experts", "input", True, "int", "", "专家总数"),
    param("communication_scope", "input", True, "string", "", "通信范围"),
    param("network_link", "input", True, "string", "", "网络链路类型"),
    param("nic_handler", "input", False, "string", "", "网卡处理器"),
    param("noise_type", "input", True, "string", "", "噪声类型"),
    param("routing_mode", "input", True, "string", "", "路由模式"),
    param("dispatch_dtype", "input", True, "string", "", "分发数据类型"),
    param("combine_dtype", "input", True, "string", "", "聚合数据类型"),
    param("dispatch_e2e_bandwidth_gbps", "output", True, "float", "GB/s", "分发端到端带宽"),
    param("dispatch_e2e_total_latency_us", "output", True, "float", "us", "分发端到端总延迟"),
    param("dispatch_send_latency_us", "output", False, "float", "us", "分发发送延迟"),
    param("dispatch_recv_latency_us", "output", False, "float", "us", "分发接收延迟"),
    param("combine_e2e_bandwidth_gbps", "output", True, "float", "GB/s", "聚合端到端带宽"),
    param("combine_e2e_total_latency_us", "output", True, "float", "us", "聚合端到端总延迟"),
    param("combine_send_latency_us", "output", False, "float", "us", "聚合发送延迟"),
    param("combine_recv_latency_us", "output", False, "float", "us", "聚合接收延迟"),
    param("dispatch_sms", "output", False, "int", "", "分发 SM 数量"),
    param("dispatch_nvl_chunk", "output", False, "int", "", "分发 NVL 分块大小"),
    param("dispatch_rdma_chunk", "output", False, "int", "", "分发 RDMA 分块大小"),
    param("dispatch_rdma_bandwidth_gbps", "output", False, "float", "GB/s", "分发 RDMA 带宽"),
    param("dispatch_nvl_bandwidth_gbps", "output", False, "float", "GB/s", "分发 NVL 带宽"),
    param("dispatch_latency_us", "output", False, "float", "us", "分发延迟"),
    param("combine_sms", "output", False, "int", "", "聚合 SM 数量"),
    param("combine_nvl_chunk", "output", False, "int", "", "聚合 NVL 分块大小"),
    param("combine_rdma_chunk", "output", False, "int", "", "聚合 RDMA 分块大小"),
    param("combine_rdma_bandwidth_gbps", "output", False, "float", "GB/s", "聚合 RDMA 带宽"),
    param("combine_nvl_bandwidth_gbps", "output", False, "float", "GB/s", "聚合 NVL 带宽"),
    param("combine_latency_us", "output", False, "float", "us", "聚合延迟"),
]


def default_config() -> Dict[str, Any]:
    hw_fields = {
        "id": None, "name": "", "address": "", "sku": "", "gpu_arch": "",
        "gpu_count": None, "gpu_core": "", "gpu_interconnect": "",
        "gpu_interconnect_bw_gbps": None, "cpu_arch": "", "cpu_kernel": "",
        "cpu_os": "", "nic_bandwidth_gbps": None, "nic_count": None,
        "nic_type": "", "nic_lanes": None, "nic_generation": "",
        "pcie_gen": "", "platform_type": "",
    }
    sw_fields = {
        "id": None, "address": "", "musa_toolkits_version": "",
        "driver_version": "", "mccl_version": "", "mtbios_version": "",
        "mtml_version": "", "mt_peermem_version": "",
    }
    return {
        "operator_identity": {
            "id": None, "name": "deepep", "family": "communication",
            "tag": None, "description": "MOE通信算子",
        },
        "hw_spec": [hw_fields],
        "sw_stack": [sw_fields],
        "operator_param_def": deepcopy(DEFAULT_OPERATOR_PARAM_DEF),
        "testcase_param_set": {
            "id": None, "operator_id": None, "tag": "deepep baseline",
            "description": "DeepEP 基准性能测试参数集",
        },
        "testcase_result": {
            "id": None, "operator_id": None, "operator_provider": "deepep",
            "operator_implementation": "deepep", "hw_spec_id": None,
            "sw_stack_id": None, "testcase_param_set_id": None,
            "operator_param_def_id": None, "value": None, "created_by": "wendy",
            "created_at": None, "started_at": None, "sw_build_at": None,
            "finished_at": None, "log_path": None, "run_status": None,
            "run_source": None, "testcase_results": [],
        },
    }


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        fail(f"JSON file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON file must contain an object: {path}")
    return value


def read_summary_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        fail(f"performance CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            fail(f"performance CSV has no header: {path}")
        fields = [field.strip() for field in reader.fieldnames]
        if any(not field for field in fields):
            fail(f"performance CSV contains an empty header field: {path}")
        rows = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                fail(f"performance CSV row does not have 36 columns at line {line_number}")
            row = {key.strip(): (value or "").strip() for key, value in raw_row.items()}
            if any(value for value in row.values()):
                rows.append(row)
    if not rows:
        fail(f"performance CSV has no data rows: {path}")
    return fields, rows


def required_schema_params(schema: Mapping[str, Any]) -> List[Dict[str, Any]]:
    params = schema.get("operator_param_def")
    if not isinstance(params, list) or not params:
        fail("embedded operator_param_def must be a non-empty list")
    normalized: List[Dict[str, Any]] = []
    for item in params:
        if not isinstance(item, dict) or not item.get("name"):
            fail("every embedded operator_param_def entry must contain name")
        normalized.append(item)
    return normalized


def as_number(value: str, data_type: str, field: str, line_number: int) -> None:
    if not value:
        return
    try:
        if data_type == "int":
            parsed = int(value)
            if str(parsed) != value and value not in {f"+{parsed}", f"-{abs(parsed)}"}:
                fail(f"line {line_number}: {field} must be an integer: {value}")
        elif data_type == "float":
            float(value)
    except ValueError:
        fail(f"line {line_number}: {field} must be {data_type}: {value}")


def validate_rows(
    schema: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
) -> None:
    if list(fields) != LATEST_COLUMNS:
        fail(
            "CSV header does not match the latest 36-column extractor schema; "
            "run extract_deepep_summary.yml first"
        )
    if len(rows) != 16:
        fail(f"CSV validation expected 16 data rows, got {len(rows)}")

    params = required_schema_params(schema)
    param_names = [str(item["name"]) for item in params]
    missing_headers = [name for name in param_names if name not in fields]
    if missing_headers:
        fail("CSV is missing config parameters: " + ", ".join(missing_headers))

    normal_fp8 = 0
    normal_bf16 = 0
    low_latency = 0
    for line_number, row in enumerate(rows, start=2):
        if row.get("primitive") != "deepep":
            fail(f"line {line_number}: primitive must be deepep")
        if row.get("implementation") not in {"deepep", "deepep_ace"}:
            fail(f"line {line_number}: implementation must be deepep or deepep_ace")
        if row.get("test_mode") not in SUPPORTED_MODES:
            fail(f"line {line_number}: unsupported test_mode: {row.get('test_mode')}")
        if row.get("combine_dtype") != "BF16":
            fail(f"line {line_number}: combine_dtype must be BF16")

        for field in NUMERIC_COLUMNS:
            value = row.get(field, "")
            if value and not NUMERIC_RE.fullmatch(value):
                fail(f"line {line_number}: numeric column {field} is invalid: {value}")
        for field in ("rank_num", "tokens", "hidden_size", "top_k", "experts"):
            if not row.get(field):
                fail(f"line {line_number}: metadata numeric column {field} is missing")
        for field in UNIFIED_COLUMNS:
            if not row.get(field):
                fail(f"line {line_number}: unified column {field} is missing")

        if row["test_mode"] == "low_latency":
            low_latency += 1
            if row.get("dispatch_dtype") != "FP8":
                fail(f"line {line_number}: low_latency dispatch_dtype must be FP8")
            for field in SEND_RECV_COLUMNS:
                if not row.get(field):
                    fail(f"line {line_number}: low_latency {field} is missing")
            for field in TUNING_COLUMNS:
                if row.get(field):
                    fail(f"line {line_number}: low_latency tuning column {field} must be empty")
        else:
            if row["test_mode"] == "intranode_normal":
                if any(row.get(field) for field in ("dispatch_rdma_chunk", "dispatch_rdma_bandwidth_gbps", "combine_rdma_chunk", "combine_rdma_bandwidth_gbps")):
                    fail(f"line {line_number}: intranode RDMA tuning columns must be empty")
            elif row["test_mode"] == "internode_normal":
                for field in ("dispatch_rdma_bandwidth_gbps", "combine_rdma_bandwidth_gbps"):
                    if not row.get(field):
                        fail(f"line {line_number}: internode {field} is missing")
            if any(row.get(field) for field in SEND_RECV_COLUMNS):
                fail(f"line {line_number}: normal send/recv latency columns must be empty")
            for field in ("dispatch_nvl_bandwidth_gbps", "dispatch_latency_us", "combine_nvl_bandwidth_gbps", "combine_latency_us"):
                if not row.get(field):
                    fail(f"line {line_number}: normal performance column {field} is missing")
            if row["dispatch_dtype"] == "FP8":
                normal_fp8 += 1
            elif row["dispatch_dtype"] == "BF16":
                normal_bf16 += 1
            else:
                fail(f"line {line_number}: normal dispatch_dtype must be FP8 or BF16")

        for item in params:
            name = str(item["name"])
            value = row.get(name, "")
            if item.get("required") and not value:
                fail(f"line {line_number}: required parameter is empty: {name}")
            as_number(value, str(item.get("data_type", "string")), name, line_number)

    if normal_fp8 != 4 or normal_bf16 != 4 or low_latency != 8:
        fail(
            "CSV validation expected 4 normal FP8, 4 normal BF16, and "
            f"8 low_latency rows; got {normal_fp8}, {normal_bf16}, {low_latency}"
        )


def unique_nonempty(rows: Iterable[Mapping[str, str]], field: str) -> List[str]:
    return sorted({row.get(field, "").strip() for row in rows if row.get(field, "").strip()})


def load_system_info(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    document = load_json(path)
    nodes = document.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        fail(f"system information must contain a non-empty nodes array: {path}")
    result: List[Dict[str, Any]] = []
    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            fail(f"system information node {index} must be an object")
        if not isinstance(node.get("hw_spec"), dict) or not isinstance(node.get("sw_stack"), dict):
            fail(f"system information node {index} must contain hw_spec and sw_stack objects")
        result.append(node)
    return result


def has_collected_value(value: Any) -> bool:
    return value is not None and value != "" and value != 0 and value != []


def find_template_node(items: Sequence[Mapping[str, Any]], node: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    address = str(node.get("address") or node.get("hw_spec", {}).get("address") or "")
    hostname = str(node.get("hostname") or node.get("hw_spec", {}).get("name") or "")
    for item in items:
        if address and str(item.get("address") or "") == address:
            return item
        if hostname and str(item.get("name") or "") == hostname:
            return item
    if index < len(items):
        return items[index]
    return items[0] if items else {}


def merge_component(base: Mapping[str, Any], collected: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in collected.items():
        if key in result and has_collected_value(value):
            result[key] = value
    return result


def merge_system_info(result: Dict[str, Any], nodes: Sequence[Mapping[str, Any]]) -> None:
    if not nodes:
        return
    template_hw = result.get("hw_spec") if isinstance(result.get("hw_spec"), list) else []
    template_sw = result.get("sw_stack") if isinstance(result.get("sw_stack"), list) else []
    hw_output: List[Dict[str, Any]] = []
    sw_output: List[Dict[str, Any]] = []
    for index, node in enumerate(nodes):
        hw = node["hw_spec"]
        sw = node["sw_stack"]
        hw_output.append(merge_component(find_template_node(template_hw, node, index), hw))
        sw_output.append(merge_component(find_template_node(template_sw, node, index), sw))
    result["hw_spec"] = hw_output
    result["sw_stack"] = sw_output


def update_config(
    schema: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    dailydate: str | None,
    log_path: str | None,
    run_status: str | None,
    run_source: str | None,
    created_by: str | None,
    operator_implementation: str | None,
    system_info: Sequence[Mapping[str, Any]],
    started_at: str | None,
    finished_at: str | None,
) -> Dict[str, Any]:
    result = deepcopy(dict(schema))
    identity = result.setdefault("operator_identity", {})
    identity.setdefault("name", "deepep")
    identity.setdefault("family", "communication")
    merge_system_info(result, system_info)

    testcase_result = result.setdefault("testcase_result", {})
    implementations = unique_nonempty(rows, "implementation")
    if operator_implementation:
        testcase_result["operator_implementation"] = operator_implementation
    elif not testcase_result.get("operator_implementation") and implementations:
        testcase_result["operator_implementation"] = "/".join(implementations)
    testcase_result["operator_provider"] = "deepep"
    testcase_result["testcase_results"] = []
    if dailydate:
        testcase_result["sw_build_at"] = dailydate
    if log_path is not None:
        testcase_result["log_path"] = log_path
    if run_status is not None:
        testcase_result["run_status"] = run_status
    if run_source is not None:
        testcase_result["run_source"] = run_source
    if created_by is not None:
        testcase_result["created_by"] = created_by
    if started_at is not None:
        testcase_result["started_at"] = started_at
    if finished_at is not None:
        testcase_result["finished_at"] = finished_at
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="DeepEP performance summary CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory where config.json is written (default: CSV directory)",
    )
    parser.add_argument(
        "--system-info",
        type=Path,
        help="system_info.json generated by scripts/collect_system_info.py",
    )
    parser.add_argument("--dailydate", help="daily build date, YYYY-MM-DD")
    parser.add_argument("--log-path", help="DeepEP test log directory")
    parser.add_argument("--run-status", default="success", help="testcase_result.run_status")
    parser.add_argument("--run-source", default="ci", help="testcase_result.run_source")
    parser.add_argument("--created-by", help="override testcase_result.created_by")
    parser.add_argument(
        "--operator-implementation",
        help="override testcase_result.operator_implementation (default: infer from CSV)",
    )
    parser.add_argument("--started-at", help="override testcase_result.started_at")
    parser.add_argument("--finished-at", help="override testcase_result.finished_at")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate CSV against the embedded schema without writing config.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        schema = default_config()
        system_info = load_system_info(args.system_info)
        fields, rows = read_summary_csv(args.csv)
        validate_rows(schema, rows, fields)
        config = update_config(
            schema,
            rows,
            dailydate=args.dailydate,
            log_path=args.log_path,
            run_status=args.run_status,
            run_source=args.run_source,
            created_by=args.created_by,
            operator_implementation=args.operator_implementation,
            system_info=system_info,
            started_at=args.started_at,
            finished_at=args.finished_at,
        )
        if args.validate_only:
            print(f"Validated {len(rows)} rows against the embedded DeepEP schema")
            return 0

        output_dir = args.output_dir or args.csv.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "config.json"
        temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temp_path.replace(output_path)
        print(f"Generated {output_path} from {len(rows)} performance rows")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
