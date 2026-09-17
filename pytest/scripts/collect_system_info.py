#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect DeepEP node hardware/software information.

The collector writes two files to ``--output-dir``:

* ``system_info.json``: structured input for ``batch_convert.py``;
* ``system_info.csv``: flat human/Jenkins inspection view.

For a two-node run, use ``--hosts 10.20.32.28,10.20.32.29``.  The local node
is collected directly and remote nodes are queried over SSH using this same
script with ``--dump-system-info-json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def run_command(command: str, timeout: int = 30) -> str:
    env = os.environ.copy()
    env["PATH"] = ":".join(
        ["/usr/local/musa/bin", "/usr/local/sbin", "/usr/sbin", "/sbin", env.get("PATH", "")]
    )
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="ignore",
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "").strip()


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def first_ipv4() -> str:
    bond = run_command("ip -4 addr show bond0 2>/dev/null")
    match = re.search(r"inet\s+([0-9.]+)/", bond)
    if match:
        return match.group(1)
    for address in run_command("hostname -I 2>/dev/null").split():
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", address) and not address.startswith("127."):
            return address
    try:
        address = socket.gethostbyname(socket.gethostname())
        return "" if address.startswith("127.") else address
    except OSError:
        return ""


def parse_section_value(text: str, section: str, key: str) -> str:
    active = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f'"{section}"') or stripped.startswith(f"'{section}'") or stripped == f"{section}:":
            active = True
            continue
        if active and stripped.startswith("}"):
            active = False
            continue
        if active and key in stripped and ":" in stripped:
            return stripped.split(":", 1)[1].strip().strip("'\" ,")
    return ""


def version_command(command: str, section: str) -> str:
    output = run_command(command)
    branch = parse_section_value(output, section, "git branch")
    commit = parse_section_value(output, section, "commit id")
    if branch and commit:
        return f"{branch}_{commit}"
    return branch or commit


def package_version(package: str) -> str:
    output = run_command(f"dpkg-query -W -f='${{Version}}' {shlex.quote(package)} 2>/dev/null")
    if output and not output.startswith("dpkg-query:"):
        return output.strip()
    output = run_command(f"dpkg -l {shlex.quote(package)} 2>/dev/null")
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and line.startswith("ii"):
            return parts[2]
    for status_path in ("/host-dpkg/status", "/host/var/lib/dpkg/status", "/var/lib/dpkg/status"):
        current = ""
        for line in read_text(status_path).splitlines():
            if line.startswith("Package: "):
                current = line.split(":", 1)[1].strip()
            elif current == package and line.startswith("Version: "):
                return line.split(":", 1)[1].strip()
    return ""


def first_regex_group(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else ""


def env_number(name: str) -> int | float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return None


def collect_cpu() -> Dict[str, str]:
    os_name = ""
    for line in read_text("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip().strip("\"'")
            break
    cpu_text = read_text("/proc/cpuinfo")
    model_match = re.search(r"(?im)^model name\s*:\s*(.+)$", cpu_text)
    model = model_match.group(1).strip() if model_match else ""
    machine = ""
    kernel = ""
    try:
        uname = os.uname()
        machine, kernel = uname.machine, uname.release
    except OSError:
        pass
    cores = os.cpu_count() or 0
    cpu_model = f"{model} ({cores} cores)" if model else kernel
    cpu_info = " | ".join(x for x in (os_name, cpu_model, machine, kernel) if x)
    return {"cpu_info": cpu_info, "cpu_arch": machine, "cpu_kernel": cpu_model, "cpu_os": os_name}


def collect_gpu() -> Dict[str, Any]:
    output = run_command("mthreads-gmi")
    sku_match = re.search(r"MTT\s+S\d+", output, re.IGNORECASE)
    sku = sku_match.group(0) if sku_match else os.environ.get("DEEPEP_GPU_SKU", "")
    count = len(re.findall(r"MTT\s+S\d+", output, re.IGNORECASE))
    if not count:
        visible = os.environ.get("MTHREADS_VISIBLE_DEVICES", "")
        count = len([x for x in visible.split(",") if x.strip()]) if visible else 0
    core_output = run_command("musaInfo 2>/dev/null | grep -i multiProcessorCount")
    if not core_output:
        core_output = run_command("muInfo 2>/dev/null | grep -i multiProcessorCount")
    cores: List[str] = []
    for line in core_output.splitlines():
        match = re.search(r"multiProcessorCount\s*[:=]?\s*(\d+)", line, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            cores.append(str(value // 2 if "S5000" in sku.upper() else value // 8))
    gpu_arch = "PH1" if "S5000" in sku.upper() else ("QY2" if "S4000" in sku.upper() else "")
    power = re.findall(r"Power Draw\s*:\s*([\d.]+)\s*W", run_command("mthreads-gmi -q -d=POWER"), re.IGNORECASE)
    return {
        "sku": sku,
        "gpu_arch": gpu_arch,
        "gpu_count": count,
        "gpu_core": "/".join(cores),
        "power_limit_W": "/".join(power),
    }


def infer_gpu_interconnect(sku: str) -> tuple[str, int | float | None]:
    interconnect = os.environ.get("DEEPEP_GPU_INTERCONNECT", "").strip()
    bandwidth = env_number("DEEPEP_GPU_INTERCONNECT_BW_GBPS")
    sku_upper = sku.upper()
    if not interconnect and ("S5000" in sku_upper or "S4000" in sku_upper):
        interconnect = "mtlink"
    if bandwidth is None and "S5000" in sku_upper:
        bandwidth = 448
    return interconnect, bandwidth


def parse_nic_lanes(entries: Iterable[str]) -> int | None:
    for entry in entries:
        match = re.search(r"\b(\d+)X\s*(?:HDR|NDR)\b", entry, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def participating_nic_count(detected_count: int, gpu_count: int, sku: str) -> int:
    configured = env_number("DEEPEP_NIC_COUNT")
    if isinstance(configured, (int, float)) and configured > 0:
        return int(configured)
    # The S5000 DeepEP topology uses one communication NIC per GPU.  Extra
    # ACTIVE mlx5 ports belong to the host but do not participate in this run.
    if "S5000" in sku.upper() and gpu_count > 0 and detected_count > gpu_count:
        return gpu_count
    return detected_count


def collect_nic() -> Dict[str, Any]:
    entries: List[str] = []
    sysfs = os.environ.get("DEEPEP_INFINIBAND_SYSFS", "/sys/class/infiniband")
    root = Path(sysfs)
    if root.is_dir():
        for device in sorted(root.glob("mlx5_*")):
            for port in ("1", "2"):
                port_dir = device / "ports" / port
                state = read_text(str(port_dir / "state"))
                phys = read_text(str(port_dir / "phys_state"))
                rate = read_text(str(port_dir / "rate"))
                if "ACTIVE" in state and "LinkUp" in phys and rate:
                    entries.append(f"{device.name}:{rate}")
    if not entries and shutil.which("ibstatus"):
        output = run_command("ibstatus")
        current = ""
        active = False
        link_up = False
        for line in output.splitlines():
            line = line.strip()
            match = re.search(r"mlx5_\d+", line)
            if match:
                current, active, link_up = match.group(0), False, False
            if "state:" in line and "ACTIVE" in line:
                active = True
            if "phys state:" in line and "LinkUp" in line:
                link_up = True
            rate = re.search(r"rate:\s*(.+)$", line)
            if rate and current and active and link_up:
                entries.append(f"{current}:{rate.group(1).strip()}")
                current = ""
    unique = list(dict.fromkeys(entries))
    rates = []
    generations = []
    for entry in unique:
        match = re.search(r"([0-9]+)\s*Gb/sec", entry)
        if match:
            rates.append(int(match.group(1)))
        gen = re.search(r"\b(NDR|HDR)\b", entry, re.IGNORECASE)
        if gen:
            generations.append(gen.group(1).upper())
    return {
        "nic_bandwidth_gbps": max(rates) if rates else None,
        "nic_count": len(unique),
        "nic_type": "ib" if unique else "",
        "nic_lanes": parse_nic_lanes(unique),
        "nic_generation": "/".join(dict.fromkeys(generations)),
        "ib_nic_info": "/".join(unique),
    }


def detect_platform_type() -> str:
    configured = os.environ.get("DEEPEP_PLATFORM_TYPE", "").strip()
    if configured:
        return configured
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return "container"
    cgroup = read_text("/proc/1/cgroup").lower()
    if any(marker in cgroup for marker in ("docker", "containerd", "kubepods", "lxc")):
        return "container"
    virtualization = run_command("systemd-detect-virt 2>/dev/null").strip().lower()
    if not virtualization or virtualization == "none":
        return "baremetal"
    return virtualization


def collect_pcie() -> str:
    output = run_command("lspci -vvv -d 1ed5: 2>/dev/null", timeout=15)
    result: List[str] = []
    for line in output.splitlines():
        if "LnkCap:" not in line or "LnkCap2" in line:
            continue
        gen = next((x for x, marker in (("5", "Gen5"), ("4", "Gen4"), ("3", "Gen3"), ("5", "32GT/s"), ("4", "16GT/s"), ("3", "8GT/s")) if marker in line), "")
        width_match = re.search(r"\bx(\d+)\b", line)
        width = width_match.group(1) if width_match else ""
        if gen and width:
            result.append(f"Gen{gen} x{width}")
        elif gen:
            result.append(f"Gen{gen}")
        elif width:
            result.append(f"x{width}")
    return "/".join(result)


def collect_local_node() -> Dict[str, Any]:
    hostname = socket.gethostname()
    address = first_ipv4()
    gpu = collect_gpu()
    nic = collect_nic()
    cpu = collect_cpu()
    gpu_interconnect, gpu_interconnect_bw = infer_gpu_interconnect(gpu["sku"])
    active_nic_count = nic["nic_count"]
    communication_nic_count = participating_nic_count(
        active_nic_count,
        gpu["gpu_count"],
        gpu["sku"],
    )
    sw = {
        "musa_toolkits_version": version_command("musa_version_query", "musa_runtime"),
        "driver_version": package_version("musa"),
        "mccl_version": version_command("mccl_version", "mccl"),
        "mtbios_version": first_regex_group(r"\d+\.\d+\.\d+", run_command("mthreads-gmi -q | grep -i MTBios")),
        "mtml_version": package_version("mtml"),
        "mt_peermem_version": package_version("mt-peermem"),
    }
    hw = {
        "id": None,
        "name": hostname,
        "address": address,
        "sku": gpu["sku"],
        "gpu_arch": gpu["gpu_arch"],
        "gpu_count": gpu["gpu_count"],
        "gpu_core": gpu["gpu_core"],
        "gpu_interconnect": gpu_interconnect,
        "gpu_interconnect_bw_gbps": gpu_interconnect_bw,
        "cpu_arch": cpu["cpu_arch"],
        "cpu_kernel": cpu["cpu_kernel"],
        "cpu_os": cpu["cpu_os"],
        "nic_bandwidth_gbps": nic["nic_bandwidth_gbps"],
        "nic_count": communication_nic_count,
        "nic_type": nic["nic_type"],
        "nic_lanes": env_number("DEEPEP_NIC_LANES") or nic["nic_lanes"],
        "nic_generation": nic["nic_generation"],
        "pcie_gen": collect_pcie(),
        "platform_type": detect_platform_type(),
    }
    return {
        "node_id": f"{hostname}_{address}" if address else hostname,
        "hostname": hostname,
        "address": address,
        "hw_spec": hw,
        "sw_stack": {"id": None, "address": address, **sw},
        "raw": {
            "cpu_info": cpu["cpu_info"],
            "ib_nic_info": nic["ib_nic_info"],
            "active_nic_count": active_nic_count,
            "nic_topo": run_command("mthreads-gmi topo -m"),
            "power_limit_W": gpu["power_limit_W"],
        },
    }


def local_aliases() -> set[str]:
    aliases = {socket.gethostname(), "localhost", "127.0.0.1", first_ipv4()}
    aliases.update(run_command("hostname -I").split())
    return {x for x in aliases if x}


def collect_nodes(hosts: Sequence[str]) -> List[Dict[str, Any]]:
    aliases = local_aliases()
    nodes: List[Dict[str, Any]] = []
    try:
        script_source = Path(__file__).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read collector source for remote execution: {exc}", file=sys.stderr)
        return []
    for host in hosts or [socket.gethostname()]:
        if host in aliases:
            nodes.append(collect_local_node())
            continue
        command = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            host, "python3", "-", "--dump-system-info-json",
        ]
        try:
            result = subprocess.run(
                command,
                input=script_source,
                capture_output=True,
                text=True,
                timeout=120,
                errors="ignore",
            )
            payload = None
            for line in reversed((result.stdout or "").splitlines()):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict) and "hw_spec" in candidate:
                        payload = candidate
                        break
                except json.JSONDecodeError:
                    continue
            if result.returncode == 0 and payload:
                nodes.append(payload)
            else:
                print(f"warning: failed to collect remote node {host}", file=sys.stderr)
        except (OSError, subprocess.TimeoutExpired):
            print(f"warning: failed to collect remote node {host}", file=sys.stderr)
    return nodes


def write_outputs(output_dir: Path, nodes: Sequence[Dict[str, Any]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "system_info.json"
    csv_path = output_dir / "system_info.csv"
    document = {
        "schema": "deepep_system_info_v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "nodes": list(nodes),
    }
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "node_id", "hostname", "address", "sku", "gpu_arch", "gpu_count", "gpu_core",
        "gpu_interconnect", "gpu_interconnect_bw_gbps", "cpu_arch", "cpu_kernel", "cpu_os",
        "nic_bandwidth_gbps", "nic_count", "nic_type", "nic_lanes", "nic_generation",
        "pcie_gen", "platform_type", "musa_toolkits_version", "driver_version", "mccl_version",
        "mtbios_version", "mtml_version", "mt_peermem_version",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for node in nodes:
            row = {"node_id": node.get("node_id", ""), "hostname": node.get("hostname", ""), "address": node.get("address", "")}
            row.update(node.get("hw_spec", {}))
            row.update(node.get("sw_stack", {}))
            writer.writerow({field: row.get(field, "") if row.get(field, "") is not None else "" for field in fields})
    return json_path, csv_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="directory for system_info.json/system_info.csv")
    parser.add_argument("--hosts", default=os.environ.get("DEEPEP_HOSTS") or os.environ.get("MCCL_HOSTS", ""), help="comma-separated host/IP list")
    parser.add_argument("--require-all-hosts", action="store_true", help="fail unless every requested host is collected")
    parser.add_argument("--dump-system-info-json", action="store_true", help="print only the local node JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.dump_system_info_json:
        print(json.dumps(collect_local_node(), ensure_ascii=False))
        return 0
    hosts = [part.split(":", 1)[0] for part in args.hosts.split(",") if part.strip()]
    nodes = collect_nodes(hosts)
    if not nodes:
        print("error: no node system information was collected", file=sys.stderr)
        return 1
    if args.require_all_hosts and hosts and len(nodes) != len(hosts):
        print(
            f"error: collected {len(nodes)} of {len(hosts)} requested host(s)",
            file=sys.stderr,
        )
        return 1
    json_path, csv_path = write_outputs(args.output_dir, nodes)
    print(f"Collected system info from {len(nodes)} node(s)")
    print(f"System info JSON: {json_path}")
    print(f"System info CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
