import argparse
from src.full_trainings import mt_training


def get_args() -> argparse.Namespace:
    """
    Instantiate argument parser and parse execution arguments

    :return: Namespace containing parsed execution arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Run trl training - machine translation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def positive_int(text: str):
        val = int(text)
        if val <= 0:
            raise argparse.ArgumentTypeError(f"{val} is not a valid value for positive integer")
        return val

    parser.add_argument(
        "-e",
        "--epochs",
        type=positive_int,
        dest="num_epochs",
        default=1,
        help="Number of training epochs. Needs to be a positive number.",
    )

    parser.add_argument(
        "-hr",
        "--human",
        dest="human_feedback",
        action="store_true",
        default=False,
        help="Run with rewards given by human.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="is_verbose",
        action="store_true",
        default=False,
        help="Show more details about execution",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    mt_training(args.human_feedback, args.is_verbose, args.num_epochs)
