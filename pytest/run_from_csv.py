#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime

# 默认日志输出目录（相对于脚本所在目录）
DEFAULT_LOG_DIR = "test_log"


def build_command(csv_path: str, server_ip: str = None, client_ip: str = None):
	"""Read commands from CSV file and build complete commands.
	
	CSV format (column order): name, mark, world_size, num_tokens, hidden, num_topk, num_experts, link_model, nic_handler, noise_mode, topk_mode, env_export, other_args, test_type
	
	Args:
		csv_path: Path to CSV file
		server_ip: Server IP address (single IP only)
		client_ip: Client IP addresses (comma-separated)
	
	Returns:
		List of dictionaries with keys: name, cmd
	"""
	# Get TEST_TYPE from environment variable
	test_type_filter = os.environ.get("TEST_TYPE_T", "").strip()
	
	# Get filter parameters from environment variables (exact match required)
	# Map environment variable names to CSV column names
	env_filters = {
		"MARK_T": "mark",
		"WORLD_SIZE_T": "world_size",
		"NUM_TOKENS_T": "num_tokens",
		"HIDDEN_T": "hidden",
		"NUM_TOPK_T": "num_topk",
		"NUM_EXPERTS_T": "num_experts",
		"NIC_HANDLER_T": "nic_handler",
		"LINK_MODEL_T": "link_model",
		"NOISE_MODE_T": "noise_mode",
		"TOPK_MODE_T": "topk_mode",
	}
	# Pre-fetch all environment variable values
	env_values = {env_var: os.environ.get(env_var, "").strip() for env_var in env_filters.keys()}
	
	# Parse command line IPs if present
	server_ip_list = []
	if server_ip:
		server_ip_list = [ip.strip() for ip in server_ip.split(",") if ip.strip()]
	
	client_ip_list = []
	if client_ip:
		client_ip_list = [ip.strip() for ip in client_ip.split(",") if ip.strip()]
	
	# If multiple server IPs are specified, skip all command generation
	if len(server_ip_list) > 1:
		print(f"Warning: Multiple server IPs specified ({len(server_ip_list)}). Skipping all command generation.")
		return []
	
	commands = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if not row:
				continue
			name = (row.get("name") or "").strip()
			if not name:
				continue
			
			# Filter by TEST_TYPE environment variable if set
			# Use contains matching: if CSV test_type contains TEST_TYPE value, it matches
			# e.g., CSV test_type="smoke-daily" matches TEST_TYPE="smoke" or "daily"
			if test_type_filter:
				row_test_type = (row.get("test_type") or "").strip()
				if test_type_filter not in row_test_type:
					continue
			
			# Filter by exact match parameters from environment variables if set
			# If environment variable is set, only process rows where CSV value exactly matches
			# If environment variable is not set, process all rows regardless of CSV value
			skip_row = False
			for env_var, csv_col in env_filters.items():
				env_value = env_values[env_var]
				if env_value:
					row_value = (row.get(csv_col) or "").strip()
					if row_value != env_value:
						skip_row = True
						break
			if skip_row:
				continue
			
			# Build command from CSV fields
			parts = []
			
			# Handle nic_handler: add environment variables at the beginning if nic_handler=cpu
			nic_handler = (row.get("nic_handler") or "").strip()
			if nic_handler == "cpu":
				parts.append("MUSA_MANAGED_FORCE_DEVICE_ALLOC=1 NVSHMEM_IBGDA_NIC_HANDLER=cpu")
			# If nic_handler=gpu or empty, do nothing
			
			# Add environment variables if present
			env_export = (row.get("env_export") or "").strip()
			if env_export:
				parts.append(env_export)
			
			# Base command
			parts.append("python run_test.py")
			
			# Add mark (required)
			mark = (row.get("mark") or "").strip()
			if not mark:
				print(f"Warning: mark is required for {name}, skipping")
				continue
			parts.append(f"-m {mark}")
			
			# Handle IP configuration based on command line arguments
			# Get world_size from CSV (always use CSV world_size)
			world_size = None
			world_size_str_csv = (row.get("world_size") or "").strip()
			if world_size_str_csv:
				try:
					world_size = int(world_size_str_csv)
				except ValueError:
					pass
			
			# If only CLIENT_IP is set without SERVER_IP, skip all commands
			if client_ip_list and not server_ip_list:
				continue
			
			# Check if world_size should be skipped based on IP configuration
			# Rule 1: If no IPs specified, skip world_size > 1
			if not server_ip_list and not client_ip_list:
				if world_size is not None and world_size > 1:
					continue
			# Rule 2: If only server_ip specified, skip world_size > 1
			elif server_ip_list and not client_ip_list:
				if world_size is not None and world_size > 1:
					continue
			# Rule 3: If server_ip and n client_ips specified, skip world_size > 1+n
			elif server_ip_list and client_ip_list:
				max_world_size = 1 + len(client_ip_list)
				if world_size is not None and world_size > max_world_size:
					continue
			
			# If SERVER_IP or CLIENT_IP is set, use command line arguments
			if server_ip_list or client_ip_list:
				# Use command line arguments for IP configuration
				
				# Helper function to add server_ip
				def add_server_ip():
					if server_ip_list:
						if len(server_ip_list) < 1:
							return False
						server_ip_val = server_ip_list[0]
						parts.append(f"--server-ip={server_ip_val}")
					return True
				
				# If world_size=1, no IPs needed
				if world_size == 1:
					# Skip all IP configuration for world_size=1
					pass
				elif world_size is not None and world_size > 1:
					# world_size > 1, need IPs
					if not add_server_ip():
						continue
					
					# Handle client_ip
					if client_ip_list:
						# CLIENT_IP is set, use command line argument
						required_client_ips = world_size - 1
						if required_client_ips > len(client_ip_list):
							continue
						
						# Take (world_size - 1) IPs from CLIENT_IP as client_ips
						client_ips_to_use = client_ip_list[:required_client_ips]
						for ip in client_ips_to_use:
							parts.append(f"--client-ip={ip}")
					else:
						# Only SERVER_IP is set, CLIENT_IP is not set
						# This should not happen due to earlier check, but keep for safety
						continue
				else:
					# world_size is not specified or None, add IPs if available
					if not add_server_ip():
						continue
					
					# Handle client_ip
					if client_ip_list:
						# Use all available client IPs
						for ip in client_ip_list:
							parts.append(f"--client-ip={ip}")
			
			# link_model 与 mark 的对应关系：
			# - internode_normal 只支持 rdma_link，且不加 --disable-nvlink
			# - intranode_normal 只有link，且支持ACE
			# - low_latency 支持 rdma_link 和 pure_rdma，仅 pure_rdma 时加 --disable-nvlink
			link_model = (row.get("link_model") or "").strip()
			mark_allowed_link = {
				"internode_normal": ["rdma_link"],
				"intranode_normal": [""],  # 不区分 rdma_link/pure_rdma，默认走 link，不添加任何 link 相关参数
				"low_latency": ["rdma_link", "pure_rdma"],
			}
			allowed = mark_allowed_link.get(mark)
			if link_model:
				if allowed is None:
					# 未知 mark，不校验 link_model，保持原逻辑：pure_rdma 加参数
					if link_model == "pure_rdma":
						parts.append("--disable-nvlink")
				elif link_model not in allowed:
					print(f"Error: {name} - mark={mark} 仅支持 link_model 为 {allowed}，当前为 '{link_model}'。跳过该条。")
					continue
				else:
					# 仅 low_latency 且 pure_rdma 时添加 --disable-nvlink
					if mark == "low_latency" and link_model == "pure_rdma":
						parts.append("--disable-nvlink")
			# internode_normal / intranode_normal 无论 link_model 为何都不加 --disable-nvlink
			
			# noise_mode 与 mark：internode_normal/intranode_normal 只支持 none，且不传 --noise-mode；
			# low_latency 支持 none、gemm、sleep、both，非 none 时传 --noise-mode=
			noise_mode = (row.get("noise_mode") or "").strip()
			mark_allowed_noise = {
				"internode_normal": ["none"],
				"intranode_normal": ["none"],
				"low_latency": ["none", "gemm", "sleep", "both"],
			}
			allowed_noise = mark_allowed_noise.get(mark)
			if allowed_noise is not None:
				noise_val = noise_mode.lower() if noise_mode else "none"
				if noise_val not in allowed_noise:
					print(f"Error: {name} - mark={mark} 仅支持 noise_mode 为 {allowed_noise}，当前为 '{noise_mode or '(空)'}'。跳过该条。")
					continue
				# internode_normal / intranode_normal 不指定 --noise-mode
				if mark == "low_latency" and noise_val != "none":
					parts.append(f"--noise-mode={noise_mode}")
			else:
				# 未知 mark，保持原逻辑
				if noise_mode and noise_mode.lower() != "none":
					parts.append(f"--noise-mode={noise_mode}")
			
			# topk_mode 与 mark：internode_normal/intranode_normal 只支持 random，且不传 --topk-mode；
			# low_latency 支持 random、uniform、no-routing，非 none 时传 --topk-mode=
			topk_mode = (row.get("topk_mode") or "").strip()
			mark_allowed_topk = {
				"internode_normal": ["random"],
				"intranode_normal": ["random"],
				"low_latency": ["random", "uniform", "no-routing"],
			}
			allowed_topk = mark_allowed_topk.get(mark)
			if allowed_topk is not None:
				topk_val = topk_mode.lower() if topk_mode else "random"
				if topk_val not in allowed_topk:
					print(f"Error: {name} - mark={mark} 仅支持 topk_mode 为 {allowed_topk}，当前为 '{topk_mode or '(空)'}'。跳过该条。")
					continue
				# internode_normal / intranode_normal 不指定 --topk-mode
				if mark == "low_latency" and topk_mode:
					parts.append(f"--topk-mode={topk_mode}")
			else:
				# 未知 mark，保持原逻辑
				if topk_mode and topk_mode.lower() != "none":
					parts.append(f"--topk-mode={topk_mode}")
			
			# Add num_tokens if present
			num_tokens = (row.get("num_tokens") or "").strip()
			if num_tokens:
				parts.append(f"--num-tokens={num_tokens}")
			
			# Add hidden and num_topk if present (both check for "none")
			# CSV column names use underscores, but command arguments use hyphens
			for csv_col, arg_name in [("hidden", "hidden"), ("num_topk", "num-topk")]:
				value = (row.get(csv_col) or "").strip()
				if value and value.lower() != "none":
					parts.append(f"--{arg_name}={value}")
			
			# Add num_experts if present
			num_experts = (row.get("num_experts") or "").strip()
			if num_experts:
				parts.append(f"--num-experts={num_experts}")
			
			# Add other_args if present
			other_args = (row.get("other_args") or "").strip()
			if other_args:
				# Check if other_args already starts with -args
				if other_args.startswith("-args"):
					parts.append(other_args)
				else:
					# Add -args prefix if not present
					parts.append(f"-args {other_args}")
			
			cmd = " ".join(parts)
			commands.append({"name": name, "cmd": cmd})
	return commands


def run_and_capture(pytest_dir: str, name: str, cmd: str, log_dir: str) -> int:
	os.makedirs(log_dir, exist_ok=True)
	log_path = os.path.join(log_dir, f"{name}.log")

	print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Starting execution: {name}")

	try:
		# Write command to log file first
		with open(log_path, "w", encoding="utf-8") as lf:
			lf.write(f"Command: {cmd}\n")
			lf.write("-" * 80 + "\n\n")
		
		# Write command output directly to log file (executed in separate process)
		with open(log_path, "a", encoding="utf-8") as lf:
			proc = subprocess.run(
				cmd,
				shell=True,
				executable="/bin/bash",
				cwd=pytest_dir,
				stdout=lf,
				stderr=subprocess.STDOUT,
				text=True,
				check=False,
			)

		# Print log to console after process ends
		print(f"\n------ {name}.log Start ------")
		try:
			with open(log_path, "r", encoding="utf-8") as rf:
				content = rf.read()
				if content.strip():
					print(content, end="" if content.endswith("\n") else "\n")
				else:
					print("(No output)")
		except Exception as read_err:
			print(f"(Failed to read log: {read_err})")
		print(f"------ {name}.log End ------\n")

		if proc.returncode != 0:  # type: ignore[attr-defined]
			print(f"\n\n\n[{name}] Exit code: {proc.returncode} (may have errors, see {log_path} for details)")  # type: ignore[attr-defined]
		else:
			print(f"\n\n\n[{name}] Execution completed")
		print("=" * 80 + "\n")
		return proc.returncode  # type: ignore[attr-defined]
	except Exception as e:
		err_msg = f"Failed to execute {name}: {e}"
		with open(log_path, "w", encoding="utf-8") as lf:
			lf.write(err_msg)
		print(err_msg)
		return 1


def main():
	parser = argparse.ArgumentParser(description="Read commands from CSV and execute them, save logs as name.log and print to console")
	parser.add_argument(
		"--csv",
		default=None,
		help="CSV configuration file path (default: DeepEP/pytest/test_cmd.csv)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Only output commands without executing them",
	)
	parser.add_argument(
		"--server-ip",
		default=None,
		help="Server IP address (single IP only, comma-separated if multiple, but multiple server IPs will skip all commands)",
	)
	parser.add_argument(
		"--client-ip",
		default=None,
		help="Client IP addresses (comma-separated)",
	)
	parser.add_argument(
		"--fail-stop",
		action="store_true",
		help="Stop after the first failing case and exit with that case's return code",
	)
	args = parser.parse_args()

	# Script directory (DeepEP/pytest)
	pytest_dir = os.path.dirname(os.path.abspath(__file__))
	default_csv = os.path.join(pytest_dir, "test_cmd.csv")
	csv_path = args.csv or default_csv

	if not os.path.exists(csv_path):
		print(f"CSV file not found: {csv_path}")
		sys.exit(1)

	try:
		cmds = build_command(csv_path, server_ip=args.server_ip, client_ip=args.client_ip)
	except Exception as e:
		print(f"Failed to read CSV: {e}")
		sys.exit(1)

	if not cmds:
		print("No valid commands found in CSV")
		sys.exit(1)
	
	total = len(cmds)
	
	# Dry run: only output commands
	if args.dry_run:
		print("\n" + "=" * 80)
		print(f"Dry run mode: {total} commands found")
		print("=" * 80)
		for i, item in enumerate(cmds, 1):
			print(f"\n[{i}] {item['name']}")
			print(f"Command: {item['cmd']}")
		print("\n" + "=" * 80)
		sys.exit(0)
	
	# 默认日志目录: pytest/test_log，不存在时自动创建，执行前清空
	log_dir = os.path.join(pytest_dir, DEFAULT_LOG_DIR)
	if os.path.exists(log_dir):
		shutil.rmtree(log_dir)
	os.makedirs(log_dir, exist_ok=True)
	
	# Execute commands
	failed = 0
	successed = 0
	for item in cmds:
		rc = run_and_capture(pytest_dir=pytest_dir, name=item["name"], cmd=item["cmd"], log_dir=log_dir)
		if rc != 0:
			failed += 1
			if args.fail_stop:
				print("\n" + "=" * 80)
				print(f"fail-stop: abort after first failure ({item['name']}) rc={rc}")
				print(f"Execution stopped: {total} planned, stopped early, {successed} successed")
				print("=" * 80)
				sys.exit(rc)
		successed += 1
	
	print("\n" + "=" * 80)
	print(f"Execution completed: {total} total, {failed} failed")
	print("=" * 80)
	
	sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
	main()

