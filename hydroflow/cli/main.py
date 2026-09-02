# -*- coding: utf-8 -*-
"""
main.py
=======
The ``hydroflow`` command-line interface.

Commands
--------
    hydroflow run -c config.yaml [--stages process_dem routing]
        Run the pipeline with the given config file (YAML, JSON or a legacy
        flat python settings module).

    hydroflow init-config [-o config.yaml]
        Write a template config file with every parameter at its default,
        ready to edit.

    hydroflow validate -c config.yaml
        Load the config file and run the pre-flight sanity checks without
        starting a simulation.

    hydroflow list-options
        Print every fixed-choice option with its integer codes (each option
        accepts the string or the code, e.g. PRECIP_METHOD: uniform ≡ 0).

The legacy ``vsa-opm`` command remains as an alias of ``hydroflow``.
"""

import argparse
import sys

from ..config import Config
from ..pipeline import run_pipeline, DEFAULT_STAGES, _STAGE_PROGRESS


def _cmd_run(args):
    cfg = Config.from_file(args.config)
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
        cfg.update_output_paths()
    if args.backend:
        cfg.BACKEND = args.backend
    cfg.validate()
    run_pipeline(cfg, stages=args.stages)
    return 0


def _cmd_init_config(args):
    cfg = Config()
    path = cfg.save(args.output)
    print(f"Template config written to: {path}")
    print("Edit at least DEM_PATH, OUTPUT_POINT, TARGET_CRS_EPSG and OUTPUT_DIR, then:")
    print(f"  hydroflow run -c {path}")
    print("Tip: fixed-choice options accept a string or an integer code "
          "(see 'hydroflow list-options').")
    return 0


def _cmd_validate(args):
    cfg = Config.from_file(args.config)
    try:
        cfg.validate()
    except ValueError as exc:
        print(exc)
        return 1
    print("Config OK.")
    return 0


def _cmd_list_options(args):
    print(Config.describe_options())
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hydroflow",
        description="hydroflow — distributed hydrological + hydrodynamic model "
                    "(VSA-OPM; Pradhan & Ogden 2010).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the pipeline from a config file")
    p_run.add_argument("-c", "--config", required=True,
                       help="config file (.yaml, .json or legacy .py)")
    p_run.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES),
                       choices=sorted(_STAGE_PROGRESS),
                       help=f"pipeline stages to run (default: {' '.join(DEFAULT_STAGES)})")
    p_run.add_argument("--output-dir", default=None,
                       help="override OUTPUT_DIR from the config file")
    p_run.add_argument("--backend", default=None, choices=("cpu", "gpu"),
                       help="override the compute backend")
    p_run.set_defaults(func=_cmd_run)

    p_init = sub.add_parser("init-config", help="write a template config file")
    p_init.add_argument("-o", "--output", default="config.yaml",
                        help="destination file (.yaml or .json; default: config.yaml)")
    p_init.set_defaults(func=_cmd_init_config)

    p_val = sub.add_parser("validate", help="check a config file without running")
    p_val.add_argument("-c", "--config", required=True,
                       help="config file (.yaml, .json or legacy .py)")
    p_val.set_defaults(func=_cmd_validate)

    p_opts = sub.add_parser("list-options",
                            help="list fixed-choice options and their integer codes")
    p_opts.set_defaults(func=_cmd_list_options)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, AttributeError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
