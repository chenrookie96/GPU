#!/usr/bin/env python3
"""
Test runner script for DeepEP tests.
Usage:
    python run_test.py -m intranode_normal       # Run intranode normal tests
    python run_test.py -m internode_normal --server-ip=<server_ip> --client-ip=<client_ip> [--client-ip=<client_ip2> ...] [--client-path=<client_path>] [-args <test_args>] # Run internode tests
    python run_test.py -m low_latency            # Run low latency tests (single node)
    python run_test.py -m low_latency -args --disable-nvlink--num-tokens=64 # Run low latency tests with additional arguments
    python run_test.py -m low_latency --server-ip=<server_ip> --client-ip=<client_ip> [--client-ip=<client_ip2> ...] # Run low latency tests (multi-node)
    python run_test.py -m low_latency --server-ip=<server_ip> --client-ip=<client_ip> --pressure-test # Run pressure test with default rounds
    python run_test.py -m low_latency --server-ip=<server_ip> --client-ip=<client_ip> --pressure-test 100 # Run pressure test with 100 rounds
    """

import subprocess
import sys
import os
import argparse
import ctypes
import signal
import time

def find_free_port() -> int:
    """Find an available TCP port on the local machine.

    Note: This is best-effort; a race is still possible if another process binds
    the port between discovery and use. It is still more reliable than choosing
    a random port.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _enable_child_subreaper() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            return
    except Exception:
        return


def _reap_children() -> int:
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        except OSError:
            break
        if pid == 0:
            break
        reaped += 1
    return reaped


def _reap_until_idle(max_wait_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        if _reap_children() == 0:
            time.sleep(0.05)
        else:
            deadline = time.monotonic() + max_wait_seconds


def _terminate_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass

def _kill_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

def run_test_multi_node(test_file, server_ip, client_ip, client_path=None, test_type="internode", pressure_test_rounds=None, test_args=None):
    """
    Run multi-node tests
    
    Args:
        test_file: Server-side test file path
        server_ip: Server IP address
        client_ip: Client IP address or list of client IP addresses
        client_path: Client path (optional)
        test_type: Test type, supports "internode" or "low_latency"
        pressure_test_rounds: If not None, run pressure test. If integer, run that many rounds.
        test_args: Additional test arguments to pass to the test file (optional)
    """
    import random
    import shlex
    
    # Ensure client_ips is a list
    if isinstance(client_ip, str):
        client_ips = [client_ip]
    else:
        client_ips = client_ip
    
    num_clients = len(client_ips)
    world_size = 1 + num_clients  # 1 for server + number of clients
    
    master_port = find_free_port()

    escaped_test_file = shlex.quote(test_file)
    
    # Determine client test filename based on test type
    if test_type == "internode":
        client_test_filename = "test_internode.py"
    elif test_type == "low_latency":
        client_test_filename = "test_low_latency.py"
    else:
        raise ValueError(f"Unsupported test type: {test_type}. Supported types: 'internode', 'low_latency'")
    
    if client_path is None:
        client_path = os.path.dirname(os.path.dirname(test_file))
        client_test_file = test_file
    else:
        client_test_file = os.path.join(client_path, "tests", client_test_filename)
    
    escaped_client_path = shlex.quote(client_path)
    escaped_client_test_file = shlex.quote(client_test_file)
    
    # Generate log file names
    master_log_file = f"master_{test_type}.log"
    client_log_files = [f"client_{test_type}_{i+1}.log" for i in range(num_clients)]
    

    
    # Build additional test args
    forwarded_test_args = []
    if test_type == "low_latency" and pressure_test_rounds is not None:
        forwarded_test_args.append("--pressure-test")
        if pressure_test_rounds > 0:
            forwarded_test_args.extend(["--pressure-test-rounds", str(pressure_test_rounds)])

    if test_args:
        # test_args can be a list (from nargs=REMAINDER) or a string
        if isinstance(test_args, list):
            forwarded_test_args.extend(test_args)
        else:
            forwarded_test_args.append(str(test_args))

    additional_test_args = ""
    if forwarded_test_args:
        quoted_args = [shlex.quote(arg) for arg in forwarded_test_args]
        additional_test_args = " " + " ".join(quoted_args)
    
    # Build environment variable export statements from current process
    env_exports = []
    env_vars_to_export = ['MUSA_MANAGED_FORCE_DEVICE_ALLOC', 'NVSHMEM_IBGDA_NIC_HANDLER']
    for env_var in env_vars_to_export:
        env_value = os.environ.get(env_var)
        if env_value:
            # Use shlex.quote to safely escape the value for shell
            escaped_value = shlex.quote(env_value)
            env_exports.append(f"export {env_var}={escaped_value}")
    
    env_export_str = " && ".join(env_exports) if env_exports else ""
    if env_export_str:
        env_export_str = env_export_str + " && "
    
    # Run server-side test (RANK=0) 
    # Export environment variables before nohup, so they are inherited by the nohup process
    # Format: export VAR=value && nohup bash -c '...'
    cmd = (
        f"{env_export_str}"
        f"nohup bash -c 'MASTER_ADDR={server_ip} MASTER_PORT={master_port} WORLD_SIZE={world_size} RANK=0 "
        f"python {escaped_test_file}{additional_test_args}' "
        f"> {master_log_file} 2>&1 &"
    )
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start server-side {test_type} test: {e}")
        return 1
    
    # Run client-side tests for each client IP
    client_ssh_list = [f"root@{client_ip}" for client_ip in client_ips]
    client_commands = []
    
    for i, (client_ip, client_ssh, client_log_file) in enumerate(zip(client_ips, client_ssh_list, client_log_files)):
        rank = i + 1  # RANK starts from 1
        escaped_client_log_file = shlex.quote(client_log_file)
        # Use full path for log file to ensure it's written in the correct location
        client_log_path = f"{client_path}/{client_log_file}"
        escaped_client_log_path = shlex.quote(client_log_path)
        # Build environment variable export statements for client
        client_env_exports = []
        for env_var in env_vars_to_export:
            env_value = os.environ.get(env_var)
            if env_value:
                escaped_value = shlex.quote(env_value)
                client_env_exports.append(f"export {env_var}={escaped_value}")
        
        client_env_export_str = " && ".join(client_env_exports) if client_env_exports else ""
        if client_env_export_str:
            client_env_export_str = client_env_export_str + " && "
        
        # Build client command with environment variables exported before nohup
        client_cmd = (
            f"cd {escaped_client_path} && "
            f"export PATH=/usr/local/musa/bin:/usr/local/mtshmem/bin:$PATH && "
            f"export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/mtshmem/lib:$LD_LIBRARY_PATH && "
            f"{client_env_export_str}"
            f"(nohup bash -c 'MASTER_ADDR={server_ip} MASTER_PORT={master_port} WORLD_SIZE={world_size} RANK={rank} "
            f"python {escaped_client_test_file}{additional_test_args}' "
            f"> {escaped_client_log_path} 2>&1 &) ; "
            f"echo 'Client {i+1} test started'"
        )
        client_commands.append((client_ssh, client_cmd, client_log_file, rank))
    
    try:
        # Start all client tests
        for client_ssh, client_cmd, client_log_file, rank in client_commands:
            result = subprocess.run(["ssh", "-n", client_ssh, client_cmd], check=True, capture_output=True, text=True, timeout=600)
            
        
        import time
        print("\nWaiting for test completion...")
        
        # Wait for test completion with timeout and periodic checks
        max_wait_time = 3600 * 3  # Maximum wait time: 3 hours
        check_interval = 10  # Check every 10 seconds
        elapsed_time = 0
        stable_count = 0  # Count consecutive checks with no log changes
        stable_threshold = 3  # Consider test complete after 3 stable checks (30s)
        
        # print(f"Will wait up to {max_wait_time} seconds for tests to complete...")
        
        # Get initial log sizes for comparison
        initial_server_log_size = 0
        initial_client_log_sizes = [0] * num_clients
        
        try:
            if os.path.exists(master_log_file):
                initial_server_log_size = os.path.getsize(master_log_file)
        except:
            pass
            
        # Get initial client log sizes
        for i, (client_ssh, _, client_log_file, _) in enumerate(client_commands):
            try:
                escaped_client_log_file = shlex.quote(client_log_file)
                client_size_cmd = f"wc -c < {escaped_client_path}/{escaped_client_log_file}"
                size_result = subprocess.run(["ssh", "-n", client_ssh, client_size_cmd], 
                                           capture_output=True, text=True, timeout=10)
                if size_result.returncode == 0:
                    initial_client_log_sizes[i] = int(size_result.stdout.strip())
            except:
                pass
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval

            # Check if all client and server processes are still running
            client_running_list = [True] * num_clients
            server_running = True
            
            # Check all client processes
            for i, (client_ssh, _, _, rank) in enumerate(client_commands):
                try:
                    check_cmd = f"pgrep -f 'python.*{client_test_filename}'"
                    result = subprocess.run(["ssh", "-n", client_ssh, check_cmd], 
                                         capture_output=True, text=True, timeout=10)
                    client_running_list[i] = result.returncode == 0
                except subprocess.TimeoutExpired:
                    print(f"Timeout checking client {rank} status at {elapsed_time}s")
                except Exception as e:
                    print(f"Error checking client {rank} status: {e}")
            
            # Check server process
            try:
                server_check_cmd = f"pgrep -f 'python.*{os.path.basename(test_file)}'"
                server_result = subprocess.run(server_check_cmd, shell=True, 
                                             capture_output=True, text=True, timeout=10)
                server_running = server_result.returncode == 0
            except subprocess.TimeoutExpired:
                print(f"Timeout checking server status at {elapsed_time}s")
            except Exception as e:
                print(f"Error checking server status: {e}")
            
            # Check log file sizes for changes
            log_changed = False
            try:
                # Check server log size
                if os.path.exists(master_log_file):
                    current_server_log_size = os.path.getsize(master_log_file)
                    if current_server_log_size != initial_server_log_size:
                        log_changed = True
                        initial_server_log_size = current_server_log_size
                
                # Check all client log sizes
                for i, (client_ssh, _, client_log_file, _) in enumerate(client_commands):
                    escaped_client_log_file = shlex.quote(client_log_file)
                    client_size_cmd = f"wc -c < {escaped_client_path}/{escaped_client_log_file}"
                    size_result = subprocess.run(["ssh", "-n", client_ssh, client_size_cmd], 
                                               capture_output=True, text=True, timeout=10)
                    if size_result.returncode == 0:
                        current_client_log_size = int(size_result.stdout.strip())
                        if current_client_log_size != initial_client_log_sizes[i]:
                            log_changed = True
                            initial_client_log_sizes[i] = current_client_log_size
            except:
                pass
            
            # Update stability counter
            if log_changed:
                stable_count = 0
            else:
                stable_count += 1
            
            # Count running processes
            num_clients_running = sum(client_running_list)
            all_clients_finished = num_clients_running == 0
            
            # Determine completion status
            if all_clients_finished and not server_running:
                print(f"All processes finished after {elapsed_time} seconds - assuming completion")
                break
            elif stable_count >= stable_threshold and (all_clients_finished or not server_running):
                print(
                    f"Processes partially finished and logs stable for {stable_count * check_interval}s "
                    f"after {elapsed_time}s - assuming completion"
                )
                break
            elif all_clients_finished:
                print(f"Waiting for server to finish...")
            elif not server_running:
                print(f"Waiting for clients to finish...")
            else:
                running_info = f"Server: {'running' if server_running else 'stopped'}, Clients: {num_clients_running}/{num_clients} running"
                print(f"Processes status: {running_info} ({elapsed_time}s elapsed)", flush=True)
        
        if elapsed_time >= max_wait_time:
            print(f"WARNING: Maximum wait time ({max_wait_time}s) reached. Proceeding to check results...")
        
        # Check test results
        print("\n\n" + "=" * 120)
        print("Checking test results")
        print("-" * 120)
        
        # Check all client logs
        for i, (client_ssh, _, client_log_file, rank) in enumerate(client_commands):
            try:
                # Get client hostname and IP
                try:
                    hostname_result = subprocess.run(["ssh", "-n", client_ssh, "hostname"], capture_output=True, text=True, timeout=10)
                    hostname_ip_result = subprocess.run(["ssh", "-n", client_ssh, "hostname -i"], capture_output=True, text=True, timeout=10)
                    client_hostname = hostname_result.stdout.strip() if hostname_result.returncode == 0 else "N/A"
                    client_hostname_ip = hostname_ip_result.stdout.strip() if hostname_ip_result.returncode == 0 else "N/A"
                    print(f"\nClient {rank} hostname: {client_hostname}")
                    print(f"Client {rank} hostname -i: {client_hostname_ip}")
                except Exception as e:
                    print(f"Warning: Cannot get client {rank} hostname: {e}")
                
                escaped_client_log_file = shlex.quote(client_log_file)
                client_log_full_path = f"{client_path}/{client_log_file}"
                escaped_client_log_full_path = shlex.quote(client_log_full_path)

                # Check log file existence/non-empty to distinguish missing/empty vs read errors
                check_cmd = f"test -s {escaped_client_log_full_path} && echo EXISTS || echo MISSING"
                check_result = subprocess.run(["ssh", "-n", client_ssh, check_cmd], capture_output=True, text=True)
                if check_result.stdout.strip() != "EXISTS":
                    ls_cmd = f"ls -la {escaped_client_log_full_path} 2>&1 || true"
                    ls_result = subprocess.run(["ssh", "-n", client_ssh, ls_cmd], capture_output=True, text=True)
                    print(f"WARNING: Client {rank} log file issue: {ls_result.stdout.strip()}")
                    return 1

                client_log_cmd = f"cat {escaped_client_log_full_path}"
                log_result = subprocess.run(["ssh", "-n", client_ssh, client_log_cmd], capture_output=True, text=True)
                if log_result.returncode == 0 and log_result.stdout.strip():
                    print(f"Client {rank} log:")
                    print(log_result.stdout)
                    
                    # Simple error check
                    if "error" in log_result.stdout.lower() or "failed" in log_result.stdout.lower():
                        print(f"ERROR: Client {rank} {test_type} test failed!")
                        return 1
                    else:
                        print(f"Client {rank} {test_type} test passed!")
                else:
                    print(f"ERROR: Cannot read client {rank} {test_type} log!")
                    return 1
            except Exception as e:
                print(f"ERROR: Cannot read client {rank} {test_type} log: {e}")
                return 1
            
        # Check server log
        try:
            # Get server hostname and IP
            try:
                hostname_result = subprocess.run(["hostname"], capture_output=True, text=True, timeout=10)
                hostname_ip_result = subprocess.run(["hostname", "-i"], capture_output=True, text=True, timeout=10)
                server_hostname = hostname_result.stdout.strip() if hostname_result.returncode == 0 else "N/A"
                server_hostname_ip = hostname_ip_result.stdout.strip() if hostname_ip_result.returncode == 0 else "N/A"
                print(f"\nServer hostname: {server_hostname}")
                print(f"Server hostname -i: {server_hostname_ip}")
            except Exception as e:
                print(f"Warning: Cannot get server hostname: {e}")
            
            with open(master_log_file, "r") as f:
                log_content = f.read()
                if log_content.strip():
                    print("\nServer log:")
                    print(log_content)
                    
                    # Simple error check
                    if "error" in log_content.lower() or "failed" in log_content.lower():
                        print(f"ERROR: Server {test_type} test failed!")
                        return 1
                    else:
                        print(f"Server {test_type} test passed!")
                else:
                    print(f"ERROR: Server {test_type} log is empty!")
                    return 1
        except FileNotFoundError:
            print(f"ERROR: Server {test_type} log file not found!")
            return 1
        except Exception as e:
            print(f"ERROR: Cannot read server {test_type} log: {e}")
            return 1
        
        print(f"\nAll {test_type} tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Failed to start client-side {test_type} test: {e}")
        return 1

def run_single_node_test(test_file, test_name, test_args=None):
    """Run a single-node test file and return the result.
    
    Args:
        test_file: Path to the test file
        test_name: Name of the test
        test_args: Additional command line arguments
    
    Note: Environment variables set in the command line (e.g., MUSA_VAR=1 python run_test.py ...)
          will be automatically inherited by the test process.
    """
    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
        return 1
    
    try:
        print(f"Running {test_name} tests from: {test_file}")
        cmd = [sys.executable, test_file]

        # Add test arguments if provided
        if test_args:
            if isinstance(test_args, list):
                cmd.extend(test_args)
            else:
                cmd.append(test_args)

        _enable_child_subreaper()
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), preexec_fn=os.setpgrp)
        try:
            proc.wait(timeout=150)
            return proc.returncode
        except subprocess.TimeoutExpired:
            print(f"Error running {test_name} tests: timed out")
            _terminate_process_group(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc.pid)
                proc.wait()
            return 1
        finally:
            _reap_until_idle()
    except Exception as e:
        print(f"Error running {test_name} tests: {e}")
        return 1

def main():
    """Run the specified tests."""
    parser = argparse.ArgumentParser(description='Run DeepEP tests')
    parser.add_argument('-m', '--mark', 
                       choices=['intranode_normal', 'low_latency', 'internode_normal'],
                       help='Test mark: intranode_normal, low_latency (single/multi-node by --server-ip/--client-ip), or internode_normal. Must be specified.')
    parser.add_argument('--server-ip', 
                       help='Server IP address for internode tests')
    parser.add_argument('--client-ip', 
                       action='append',
                       help='Client IP address for internode tests (can be specified multiple times for multiple clients)')
    parser.add_argument('--client-path', 
                       help='Client path for internode tests')
    parser.add_argument('-p', '--pressure-test', 
                       nargs='?', 
                       const=0, 
                       type=int,
                       dest='pressure_test',
                       help='Run pressure test. If specified without value, run with default rounds. If specified with integer value (e.g., --pressure-test 100), run that many rounds.')
    parser.add_argument('-args', '--test-args',
                       dest='test_args',
                       nargs=argparse.REMAINDER,
                       help='Additional test arguments to pass to the test file (e.g., -args --num-tokens=50 --disable-nvlink or -args --disable-nvlink)')
    # Use parse_known_args to allow unrecognized arguments (which will be passed to test files)
    args, unknown_args = parser.parse_known_args()
    
    # Merge unknown arguments with test_args (unknown_args before -args, test_args after -args)
    # Note: When using nargs=REMAINDER, all arguments after -args are collected into test_args
    # unknown_args only contains unrecognized arguments before -args
    # If -args is used, args.test_args will be a list (empty list if no args after -args)
    # If -args is not used, args.test_args will be None
    if unknown_args:
        # If there are unknown args before -args, merge them with test_args
        if args.test_args is None:
            args.test_args = unknown_args
        else:
            args.test_args = unknown_args + args.test_args
    # If args.test_args is None (not used), keep it as None
    # If args.test_args is [] (used but no args), keep it as [] (will be handled correctly in functions)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Update tests directory path
    tests_dir = os.path.join(script_dir, "..", "tests-v1")
    
    # Mark must be specified
    if args.mark is None:
        print("Error: Test mark (-m/--mark) must be specified.")
        print("Available test marks:")
        print("  - intranode_normal: Run intranode normal tests")
        print("  - low_latency: Run low latency tests (single node without IPs, multi-node with --server-ip and --client-ip)")
        print("  - internode_normal: Run internode normal tests (requires --server-ip and --client-ip)")
        print("\nUsage examples:")
        print("  python run_test.py -m intranode_normal")
        print("  python run_test.py -m low_latency")
        print("  python run_test.py -m low_latency --server-ip=<ip> --client-ip=<ip>")
        print("  python run_test.py -m internode_normal --server-ip=<ip> --client-ip=<ip>")
        sys.exit(1)
    
    # Run specific test based on mark
    if args.mark == 'intranode_normal':
        test_file = os.path.join(tests_dir, "test_intranode.py")
        result = run_single_node_test(test_file, "intranode", args.test_args)
    elif args.mark == 'low_latency':
        test_file = os.path.join(tests_dir, "test_low_latency.py")
        if args.server_ip and args.client_ip:
            # Multi-node: internode low latency logic
            result = run_test_multi_node(test_file, args.server_ip, args.client_ip, args.client_path, "low_latency", args.pressure_test, args.test_args)
        else:
            # Single node: intranode low latency logic
            result = run_single_node_test(test_file, "low_latency", args.test_args)
    elif args.mark == 'internode_normal':
        if not args.server_ip or not args.client_ip:
            print("Error: internode tests require --server-ip and --client-ip arguments")
            print("Usage: python run_test.py -m internode_normal --server-ip=<ip> --client-ip=<ip> [--client-ip=<ip2> ...] [--client-path=<path>] [-args <test_args>]")
            sys.exit(1)
        test_file = os.path.join(tests_dir, "test_internode.py")
        result = run_test_multi_node(test_file, args.server_ip, args.client_ip, args.client_path, "internode", None, args.test_args)
    sys.exit(result)

if __name__ == "__main__":
    main()
