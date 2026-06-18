from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test sederhana untuk Fraud Detection API.")
    parser.add_argument("--mode", choices=["individual", "batch"], default="individual")
    parser.add_argument("--requests", type=int, default=100, help="Jumlah transaksi/request yang diuji.")
    parser.add_argument("--concurrency", type=int, default=20, help="Jumlah request paralel untuk mode individual.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL FastAPI.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout request dalam detik.")
    parser.add_argument("--seed", type=int, default=42, help="Seed random agar hasil bisa diulang.")
    return parser.parse_args()


def make_transaction() -> dict[str, float | int]:
    return {
        "distance_from_home": round(random.uniform(0, 150), 2),
        "distance_from_last_transaction": round(random.uniform(0, 60), 2),
        "ratio_to_median_purchase_price": round(random.uniform(0.1, 12), 2),
        "repeat_retailer": random.randint(0, 1),
        "used_chip": random.randint(0, 1),
        "used_pin_number": random.randint(0, 1),
        "online_order": random.randint(0, 1),
    }


def post_json(url: str, payload: Any, timeout: float) -> tuple[bool, float, Any]:
    started = time.perf_counter()
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            elapsed = time.perf_counter() - started
            return True, elapsed, json.loads(body)
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        return False, elapsed, f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except URLError as exc:
        elapsed = time.perf_counter() - started
        return False, elapsed, f"URL error: {exc.reason}"
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return False, elapsed, f"{type(exc).__name__}: {exc}"


def run_individual(base_url: str, total_requests: int, concurrency: int, timeout: float) -> None:
    url = f"{base_url.rstrip('/')}/predict"
    transactions = [make_transaction() for _ in range(total_requests)]

    started = time.perf_counter()
    results: list[tuple[bool, float, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(post_json, url, transaction, timeout) for transaction in transactions]
        for future in as_completed(futures):
            results.append(future.result())

    print_summary(results, time.perf_counter() - started)


def run_batch(base_url: str, total_requests: int, timeout: float) -> None:
    url = f"{base_url.rstrip('/')}/predict-batch"
    transactions = [make_transaction() for _ in range(total_requests)]

    started = time.perf_counter()
    result = post_json(url, transactions, timeout)
    print_summary([result], time.perf_counter() - started)

    success, _, payload = result
    if success:
        print(f"Total transaksi : {payload.get('total')}")
        print(f"Fraud           : {payload.get('fraud')}")
        print(f"Non-Fraud       : {payload.get('non_fraud')}")


def print_summary(results: list[tuple[bool, float, Any]], total_elapsed: float) -> None:
    success_count = sum(1 for success, _, _ in results if success)
    failed_count = len(results) - success_count
    latencies = [elapsed for success, elapsed, _ in results if success]

    print(f"Total request   : {len(results)}")
    print(f"Sukses          : {success_count}")
    print(f"Gagal           : {failed_count}")
    print(f"Total waktu     : {total_elapsed:.3f} detik")

    if total_elapsed > 0:
        print(f"Throughput      : {len(results) / total_elapsed:.2f} request/detik")

    if latencies:
        print(f"Latency rata2   : {statistics.mean(latencies) * 1000:.2f} ms")
        print(f"Latency median  : {statistics.median(latencies) * 1000:.2f} ms")
        print(f"Latency min/max : {min(latencies) * 1000:.2f} / {max(latencies) * 1000:.2f} ms")

    if failed_count:
        print("\nContoh error:")
        for success, _, payload in results:
            if not success:
                print(payload)
                break


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.requests <= 0:
        raise SystemExit("--requests harus lebih dari 0.")

    if args.mode == "individual":
        if args.concurrency <= 0:
            raise SystemExit("--concurrency harus lebih dari 0.")
        run_individual(args.base_url, args.requests, args.concurrency, args.timeout)
    else:
        run_batch(args.base_url, args.requests, args.timeout)


if __name__ == "__main__":
    main()
