import csv
import time
import numpy as np
SIZES = [128, 256, 512, 1024]
REPEATS = 20

def benchmark_matvec(n, dtype, repeats=REPEATS):
    A = np.full((n, n), 1.0, dtype=dtype)
    x = np.full(n, 3.0, dtype=dtype)
    y = np.empty(n, dtype=dtype)

    # Run it once first as a warm-up; don’t include it in the timing
    np.matmul(A, x, out=y)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.matmul(A, x, out=y)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times)

def benchmark_matmul(n, dtype, repeats=REPEATS):
    A = np.full((n, n), 1.0, dtype=dtype)
    B = np.full((n, n), 2.0, dtype=dtype)
    C = np.empty((n, n), dtype=dtype)

    # Run it once first as a warm-up; don’t include it in the timing
    np.matmul(A, B, out=C)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.matmul(A, B, out=C)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times)

if __name__ == "__main__":
    results = []
   
    for dtype in (np.float32, np.float64):
        print(f"\nData types:：{np.dtype(dtype).name}")

        for n in SIZES:

            matvec_time = benchmark_matvec(n, dtype)
            matvec_flops = 2 * n**2
            matvec_gflops = matvec_flops / matvec_time / 1e9

            bytes_per_number = np.dtype(dtype).itemsize
            matvec_bytes = bytes_per_number * (n**2 + 2 * n)
            matvec_intensity = matvec_flops / matvec_bytes
            matvec_bandwidth = matvec_bytes / matvec_time / 1e9

            matmul_time = benchmark_matmul(n, dtype)
            matmul_flops = 2 * n**3
            matmul_gflops = matmul_flops / matmul_time / 1e9
            matmul_bytes = bytes_per_number * (3 * n**2)
            matmul_intensity = matmul_flops / matmul_bytes
            matmul_bandwidth = matmul_bytes / matmul_time / 1e9

            print(f"Matvec: {matvec_intensity:.3f} FLOP/byte, "
                f"{matvec_bandwidth:.3f} GB/s, "
                f"{matvec_gflops:.3f} GFLOP/s")

            print(f"Matmul: {matmul_intensity:.3f} FLOP/byte, "
                f"{matmul_bandwidth:.3f} GB/s, "
                f"{matmul_gflops:.3f} GFLOP/s")

            print(
                f"N={n}: "
                f"MatVec time={matvec_time:.6f} s, "
                f"MatMul time={matmul_time:.6f} s"
            )
            results.append({
                    "dtype": np.dtype(dtype).name,
                    "N": n,
                    "matvec_time_s": matvec_time,
                    "matvec_flops": matvec_flops,
                    "matvec_gflops": matvec_gflops,
                    "matvec_bytes": matvec_bytes,
                    "matvec_intensity_flop_per_byte": matvec_intensity,
                    "matvec_bandwidth_gb_s": matvec_bandwidth,
                    "matmul_time_s": matmul_time,
                    "matmul_flops": matmul_flops,
                    "matmul_gflops": matmul_gflops,
                    "matmul_bytes": matmul_bytes,
                    "matmul_intensity_flop_per_byte": matmul_intensity,
                    "matmul_bandwidth_gb_s": matmul_bandwidth,
                })
    with open(
            "4_Numerical_precision_and_stability/results.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)