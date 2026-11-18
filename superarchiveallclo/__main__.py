"""Generate a prototype sewing pattern from a tee parameter JSON file."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .export import write_exports
from .tee import TeeSpec, draft


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("parameters", type=Path, help="Tee measurements in millimetres")
    parser.add_argument("-o", "--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--variants", action="store_true", help="Also generate three one-parameter variants")
    args = parser.parse_args()

    try:
        parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
        if not isinstance(parameters, dict):
            raise ValueError("Parameter JSON must be an object")
        spec = TeeSpec(**parameters)
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as temporary:
            bundle = Path(temporary) / "bundle"
            if args.variants:
                bundle.mkdir()
                if spec.garment_shoulder_width_mm is not None:
                    dropped = replace(spec, garment_shoulder_width_mm=spec.shoulder_width_mm + 20)
                else:
                    dropped = replace(spec, shoulder_extension_mm=spec.shoulder_extension_mm + 10)
                variants = {
                    "base": spec,
                    "shorter-body": replace(spec, back_length_mm=spec.back_length_mm - 20),
                    "wider-chest": replace(spec, flat_chest_mm=spec.flat_chest_mm + 20),
                    "more-drop-shoulder": dropped,
                }
                for name, variant in variants.items():
                    write_exports(draft(variant), bundle / name)
            else:
                write_exports(draft(spec), bundle)
            bundle.rename(args.output)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
