import re
import sys


def parse_log_file(file_path):
    # Initialize results dictionary
    results = {i: {"passed": False, "tflops": None} for i in range(1, 11)}

    try:
        with open(file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Pattern to match the full test line with TFLOPS
    test_line_pattern = re.compile(r"tests/test_step(\d+)\.py::test_.*?(\d+\.?\d*) TFLOP/S")

    # Pattern to match PASSED status
    passed_pattern = re.compile(r"PASSED")

    # Pattern to match step execution messages
    step_execution_pattern = re.compile(
        r"Running step (\d+):|"
        r"Step (\d+) passed\."
    )

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
            step_num = int(test_lines[0][0])

            # Check if any test in this block PASSED
            if passed_pattern.search(block):
                results[step_num]["passed"] = True

                # Extract the maximum TFLOPS from all subtests in this step
                tflops_values = []
                for match in test_line_pattern.finditer(block):
                    try:
                        tflops = float(match.group(2))
                        tflops_values.append(tflops)
                    except ValueError:
                        continue

                if tflops_values:
                    # Use the maximum TFLOPS value (or you could use average)
                    results[step_num]["tflops"] = max(tflops_values)

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
        result = results[step_num]
        status = "PASSED" if result["passed"] else "FAILED"

        if result["passed"]:
            tflops_str = f"{result['tflops']:.2f} TFLOP/S" if result["tflops"] is not None else "N/A"
            print(f"Step {step_num:2d}: {status} ({tflops_str})")
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
