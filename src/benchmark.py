
import time
import json
import platform
from pathlib import Path

def run_benchmark(size, repeats=3):
    """Benchmark a lightweight CPU workload."""
    a = list(range(size))
    results = []

    for _ in range(repeats):
        start = time.perf_counter()

        total = sum(x * x for x in a)

        elapsed = time.perf_counter() - start
        results.append(elapsed)

    return {
        "workload_size": size,
        "repeats": repeats,
        "average_seconds": round(sum(results) / len(results), 6),
        "result_checksum": total
    }


def main():
    print("=== GenAI Inference Optimizer ===")
    print("Benchmark mode: CPU baseline")
    print("Platform:", platform.platform())

    results = [
        run_benchmark(10_000),
        run_benchmark(50_000),
        run_benchmark(100_000)
    ]

    for result in results:
        print(json.dumps(result, indent=2))

    output_path = Path("benchmark_results.json")
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

