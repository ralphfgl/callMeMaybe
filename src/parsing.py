from argparse import ArgumentParser, Namespace


def get_arguments() -> Namespace:
    """Initialize the command line arguments for the program and returns them in a Namespace object"""

    parser = ArgumentParser(
        prog="python3 -m src",
        description="CallMeMaybe 42 project done by rfeghali",
        epilog="I'll guess we'll never know",
    )
    parser.add_argument(
        "-d",
        "--functions_definition",
        help="Path of functions definition file.",
        default="data/input/functions_definition.json",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path of the input file (prompt).",
        default="data/input/function_calling_tests.json",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/output/function_calling_result.json",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model.",
        default="Qwen/Qwen3-0.6B",
        required=False,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Execution device",
    )
    return parser.parse_args()
