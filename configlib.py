"""Configuration for experiments.
Taken from https://github.com/nuric/pix2rule"""
from typing import Any, Dict, Callable
import argparse
import json
import logging

import utils.hashing

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Global configuration dict
config: Dict[str, Any] = {}


def add_group(
    title: str, prefix: str = "", description: str = ""
) -> Callable[..., Any]:
    """Create a new context for arguments and return a handle."""
    arg_group = parser.add_argument_group(title, description)
    prefix = prefix + "_" if prefix else prefix

    def add_argument_wrapper(name: str, **kwargs: Any):
        """Wrapper for adding arguments into the group."""
        # Strip -- if exists
        name = name.removeprefix("--")
        # Add splitting character
        arg_group.add_argument("--" + prefix + name, **kwargs)

    return add_argument_wrapper


def add_arguments_dict(
    add_function: Callable[..., Any],
    arguments: Dict[str, Dict[str, Any]],
    prefix: str = "",
) -> None:
    """Add arguments from a dictionary into the parser with given prefix."""
    prefix = prefix + "_" if prefix else prefix
    for argname, conf in arguments.items():
        add_function(prefix + argname, **conf)


def parse(save_fname: str = "") -> str:
    """Clean configuration and parse given arguments."""
    # Start from clean configuration
    config.clear()
    config.update(vars(parser.parse_args()))
    logger.info("Parsed %i arguments.", len(config))
    # Save passed arguments
    if save_fname:
        save_config(save_fname)
    return utils.hashing.dict_hash(config)


def save_config(filename: str = "config.json") -> None:
    """Save config file as a json."""
    assert filename.endswith(
        ".json"
    ), f"Config file needs end with json, got {filename}."
    with open(filename, "w", encoding="utf8") as config_file:
        json.dump(config, config_file, indent=4)
    logger.info("Saved configuration to %s.", filename)
