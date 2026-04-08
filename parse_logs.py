import re
import sys

# from lec slides https://mlsyscourse.org/slides/tirx-gemm/#/35
BENCHMARK4096 = [0.02, 0.02, 3, 330, 639, 723, 603, 1057, 1238, 1322]


def parse_log_file(file_path):
    # Initialize results dictionary
    results = {i: {"passed": False, "perf": {}} for i in range(1, 11)}

    try:
        with open(file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Pattern to match the full test line with TFLOPS
    # https://regex101.com/r/T4ZSoH/1
    # test_line_pattern = re.compile(r"tests\/test_step(\d+)\.py::test_k_loop\[(\d+)\] .*? (\d+\.?\d*) TFLOP\/S")
    test_line_pattern = re.compile(r"tests\/test_step(\d+)\.py::test_.*?\[(\d+)\] .*? (\d+\.?\d*) TFLOP\/S")

    # Pattern to match PASSED status
    passed_pattern = re.compile(r"PASSED")

    # # Pattern to match step execution messages
    # step_execution_pattern = re.compile(
    #     r"Running step (\d+):|"
    #     r"Step (\d+) passed\."
    # )

    # Pattern to match test session starts (to separate test blocks)
    session_pattern = re.compile(r"============================= test session starts ==============================")

    # Split content into blocks based on test session starts
    blocks = session_pattern.split(content)

    # Process each block (skip the first empty one)
    for i, block in enumerate(blocks[1:], 1):
        if not block.strip():
            continue

        # Find all test lines in this block
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

    # Print results
    print("Step Results:")
    print("-" * 40)

    all_passed = True
    for step_num in range(1, 11):
        print("-----")
        result = results[step_num]
        status = "PASSED" if result["passed"] else "FAILED"

        if result["passed"]:
            # print(result["perf"])
            for size, tflops in result["perf"].items():
                tflops_str = f"size = {size}, {tflops} TFLOP/S" if result["perf"][size] is not None else "N/A"

                print(f"Step {step_num:2d}: {status} ({tflops_str})")
            
            benchmark = BENCHMARK4096[step_num - 1]
            try:
                score4096 = float(result["perf"]["4096"])
                if score4096 >= benchmark:
                    print(f"Passes Benchmark for (size = 4096), since score of {score4096} TFLOP/S exceeds benchmark {benchmark} TFLOP/S")
                else:
                    print(f"FAILS Benchmark for (size = 4096), since score of {score4096} TFLOP/S lower than benchmark {benchmark} TFLOP/S")
            except KeyError:
                print(f"Test was not run with size=4096 on Modal, benchmark is {benchmark} TFLOP/S")

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
