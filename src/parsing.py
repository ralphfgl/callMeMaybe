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
        type=str,
        default="data/input/functions_definition.json",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path of the input file (prompt).",
        type=str,
        default="data/input/function_calling_tests.json",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/output/function_calling_result.json",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the model.",
        default="Qwen/Qwen3-0.6B",
        required=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Execution device",
    )
    return parser.parse_args()


# def parse_arguments() -> Namespace:
#     """Parse command line arguments for the application."""
#     parser = ArgumentParser(
#         prog="CallMeMaybe",
#         description="Call Me Maybe: LLM Function Calling Tool",
#         usage="uv run python -m src [--functions_definition "
#         "<function_definition_file>] [--input <input_file>] "
#         "[--output <output_file>]",
#         epilog="I'll guess we will never now",
#     )
#     parser.add_argument(
#         "--functions_definition",
#         metavar="",
#         type=str,
#         default="data/input/functions_definition.json",
#         help="Path to JSON file containing the functions definitions.",
#     )
#     parser.add_argument(
#         "--input",
#         metavar="",
#         type=str,
#         default="data/input/function_calling_tests.json",
#         help="Path to the file containing the prompts.",
#     )
#     parser.add_argument(
#         "--output",
#         metavar="",
#         type=str,
#         default="data/output/function_calling_results.json",
#         help="Path to the JSON output file.",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="Qwen/Qwen3-0.6B",
#         help="HuggingFace Model ID",
#     )
#     args = parser.parse_args()
#     return args
