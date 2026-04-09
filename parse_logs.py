import re
import sys

# from lec slides https://mlsyscourse.org/slides/tirx-gemm/#/35
SIZES = ["default", "4096"] + (["2048"] * 2) + (["4096"] * 7)
# the slides lied...
BENCHMARKS = [0.02, 0.02, 3, 330, 505.29, 723, 603, 1057, 1238, 1322]


def parse_log_file(file_path):
    # Initialize results dictionary
    results = {i: {"passed": False, "perf": {}} for i in range(1, 11)}

    try:
        with open(file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Pattern to match test session starts (to separate test blocks)
    session_pattern = re.compile(r"============================= test session starts ==============================")

    # Split content into blocks based on test session starts
    blocks = session_pattern.split(content)

    # =============================
    # Process First Block (expect 0.02 TFLOP/S)
    # =============================
    def process_test_1(block):
        test1_pattern = re.compile(r"tests\/test_step(\d+)\.py::test_.*? (\d+\.?\d*) TFLOP\/S")
        test_lines = test1_pattern.findall(block)

        # Find step number from the first test line
        if test_lines:
            for test_line in test_lines:
                step_num, tflops = test_line
                print(step_num, tflops)
                step_num = int(step_num)
                results[step_num]["perf"]["default"] = tflops

        # Also check for "Step X passed" messages
        step_passed_matches = re.findall(r"Step (\d+) passed\.", block)
        for match in step_passed_matches:
            step_num = int(match)
            results[step_num]["passed"] = True

    # =============================
    # https://regex101.com/r/T4ZSoH/1
    test_line_pattern = re.compile(r"tests\/test_step(\d+)\.py::test_.*?\[(\d+)\] .*? (\d+\.?\d*) TFLOP\/S")

    # =============================
    # Process Second Block (expect 0.02 TFLOP/S) at 128 x 4096
    # =============================
    def process_test(test_idx, target_size, benchmark_tflops):
        test_lines = test_line_pattern.findall(block)

        # Find step number from the first test line
        if test_lines:
            for test_line in test_lines:
                step_num, size, tflops = test_line
                step_num = int(step_num)
                results[step_num]["perf"][size] = tflops

        # Also check for "Step X passed" messages
        step_passed_matches = re.findall(r"Step (\d+) passed\.", block)
        for match in step_passed_matches:
            step_num = int(match)
            results[step_num]["passed"] = True

    # Process each block (skip the first empty one)
    for i, block in enumerate(blocks[1:], 1):
        if not block.strip():
            continue

        if i == 0:
            process_test_1(block)
        elif i == 1 or i == 2:
            process_test(i + 1, "2048", BENCHMARKS[i])
        else:
            process_test(i + 1, "4096", BENCHMARKS[i])

    # Print results
    print("Step Results:")
    print("-" * 40)

    all_passed = True
    for step_num in range(1, 11):
        print("-----")
        result = results[step_num]
        status = "PASSED" if result["passed"] else "FAILED"

        if result["passed"]:
            for size, tflops in result["perf"].items():
                tflops_str = f"size = {size}, {tflops} TFLOP/S" if result["perf"][size] is not None else "N/A"

                print(f"Step {step_num:2d}: {status} ({tflops_str})")

            benchmark = BENCHMARKS[step_num - 1]
            size = SIZES[step_num - 1]
            try:
                score = float(result["perf"][size])
                if score >= benchmark:
                    print(f"Passes Benchmark for (size = {size}), since score of {score} TFLOP/S exceeds benchmark {benchmark} TFLOP/S")
                elif score / benchmark > 0.9:
                    print(f"Close to Benchmark for (size = {size}), since score of {score} TFLOP/S exceeds 90% of benchmark {benchmark} TFLOP/S")
                else:
                    print(f"FAILS Benchmark for (size = {size}), since score of {score} TFLOP/S lower than benchmark {benchmark} TFLOP/S")
            except KeyError:
                print(f"Test was not run with size={size} on Modal, benchmark is {benchmark} TFLOP/S")
                print(result["perf"])
                print(results)

        else:
            print(f"Step {step_num:2d}: {status}")

        if not result["passed"]:
            all_passed = False

    print("-" * 40)
    if all_passed:
        print("All steps passed!")
    else:
        print("Some steps failed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_log.py <log_file_path>")
        sys.exit(1)

    parse_log_file(sys.argv[1])
