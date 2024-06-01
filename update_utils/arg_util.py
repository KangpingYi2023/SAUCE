import argparse
from enum import Enum, auto
from typing import List


class ArgType(Enum):
    DATA_UPDATE = auto()
    DATASET = auto()
    DEBUG = auto()
    DRIFT_TEST = auto()
    END2END = auto()
    EVALUATION_TYPE = auto()
    MODEL_UPDATE = auto()


DICT_FROM_ARG_TO_ALLOWED_ARG_VALS = {
    ArgType.DATASET: ["census", "forest", "bjaq", "power"],
    ArgType.EVALUATION_TYPE: ["estimate", "drift"],
}


def add_common_arguments(parser: argparse.ArgumentParser, arg_types: List[ArgType]):
    for arg_type in arg_types:
        if arg_type == ArgType.DATA_UPDATE:
            parser.add_argument(
                "--data_update",
                type=str,
                choices=["permute-opt", "permute", "sample", "single", "value", "tupleskew", "valueskew"],
                help="permute (DDUp), sample (FACE), permute (FACE), single (our)",
            )
            parser.add_argument(
                "--update_size", type=int, default=20000, help="default=20000"
            )
            parser.add_argument("--num_queries", type=int, default=1000, help="# queries.")
        if arg_type == ArgType.DATASET:
            parser.add_argument(
                "--dataset",
                type=str,
                choices=["bjaq", "census", "forest", "power"],
                required=True,
                help="choose datasets: bjaq, census, forest, power",
            )
        if arg_type == ArgType.DEBUG:
            parser.add_argument(
                "--debug",
                action="store_true", 
                default=False,
            )
        if arg_type == ArgType.DRIFT_TEST:
            parser.add_argument(
                "--drift_test",
                type=str,
                choices=["sauce", "ddup", "none"],
                help="sauce, ddup, none",
            )
        if arg_type == ArgType.END2END:
            parser.add_argument(
                "--end2end",
                action="store_true", 
                default=False,
                help="activate end2end experiment",
            )
            parser.add_argument(
                "--query_seed",
                type=int,
                default=1234,
            )
        if arg_type == ArgType.EVALUATION_TYPE:
            parser.add_argument(
                "--eval_type",
                type=str,
                choices=["estimate", "drift"],
                required=True,
                help="estimate, drift",
            )
        if arg_type == ArgType.MODEL_UPDATE:
            parser.add_argument(
                "--model_update",
                type=str,
                choices=["update", "adapt", "finetune", "none"],
                help="update (drift_test=ddup), adapt (drift_test=sauce), finetune (baseline), none (no updates)",
            )
            parser.add_argument(
                "--model",
                type=str,
                choices=["naru", "face", "transformer"],
                help="naru, face",
            )
            



def validate_argument(arg_type: ArgType, arg_val: str):
    """
    Validates if the provided argument is allowed.

    Args:
        arg_type (ArgType): The type of the argument to be validated.
        arg_val (str): The value of the argument to be validated.

    Raises:
        ValueError: If the argument value is not in the allowed list.
    """
    if arg_val not in DICT_FROM_ARG_TO_ALLOWED_ARG_VALS.get(arg_type, []):
        raise ValueError(f'Validate Argument: UNKNOWN {arg_type.name}="{arg_val}"')
    print(f'Validate Argument: {arg_type.name}="{arg_val}" is valid.')


if __name__ == "__main__":
    # validate_argument(ArgType.DATASET, "census")
    # validate_argument(ArgType.EVALUATION_TYPE, "estimate")
    pass
