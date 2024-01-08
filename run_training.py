import argparse
from src.training import TranslationTraining, ReviewTraining

TYPE_CHOICES = ["translation", "review"]


def get_args() -> argparse.Namespace:
    """
    Instantiate argument parser and parse execution arguments

    :return: Namespace containing parsed execution arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Run trl training - translation or review task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--type",
        choices=TYPE_CHOICES,
        default=TYPE_CHOICES[0],
        dest="type",
        help="Choose training task.",
    )

    def positive_int(text: str) -> int:
        val = int(text)
        if val <= 0:
            raise argparse.ArgumentTypeError(f"{val} is not a valid value for positive integer.")
        return val

    parser.add_argument(
        "-e",
        "--epochs",
        type=positive_int,
        dest="epochs",
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
        dest="debug",
        action="store_true",
        default=False,
        help="Show more details about execution",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.type == "translation":
        trainer = TranslationTraining(args.human_feedback, args.debug, args.epochs)
    else:
        trainer = ReviewTraining(args.human_feedback, args.debug, args.epochs)
    trainer.train()


if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
