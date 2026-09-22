from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from projectkoios.agent.organizer.catalog import OrganizerCatalog
from projectkoios.agent.organizer.classification import (
    OllamaMetadataCategorizer,
)
from projectkoios.agent.organizer.daemon import OrganizerDaemon
from projectkoios.agent.organizer.models import OrganizerControlMode


def catalog_path() -> Path:
    explicit = os.environ.get("KOIOS_ORGANIZER_CATALOG")
    if explicit:
        return Path(explicit).expanduser()
    data_root = Path(
        os.environ.get(
            "KOIOS_DATA_ROOT",
            "~/projectkoios/.koios/store-v1",
        )
    ).expanduser()
    return data_root / "state" / "organizer" / "catalog.sqlite3"


def control_main() -> None:
    parser = argparse.ArgumentParser(prog="koios-organizer")
    parser.add_argument("command", choices=("on", "pause", "off", "status"))
    arguments = parser.parse_args()
    catalog = OrganizerCatalog(catalog_path())
    if arguments.command != "status":
        catalog.set_mode(OrganizerControlMode(arguments.command))
    status = catalog.status()
    print(
        json.dumps(
            {
                "desired_mode": status.desired_mode.value,
                "activity": status.activity.value,
                "discovered_roots": status.discovered_roots,
                "observed_files": status.observed_files,
                "local_files": status.local_files,
                "placeholder_files": status.placeholder_files,
                "proposed_files": status.proposed_files,
                "last_event_sequence": status.last_event_sequence,
                "current_root_id": status.current_root_id,
                "current_relative_path": status.current_relative_path,
                "last_error": status.last_error,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def daemon_main() -> None:
    model = os.environ.get("KOIOS_ORGANIZER_MODEL", "qwen3.5:9b")
    model_digest = os.environ.get("KOIOS_ORGANIZER_MODEL_DIGEST")
    if model_digest is None:
        raise SystemExit("KOIOS_ORGANIZER_MODEL_DIGEST is required")
    catalog = OrganizerCatalog(catalog_path())
    categorizer = OllamaMetadataCategorizer(
        model=model,
        model_digest=model_digest,
    )
    OrganizerDaemon(catalog, categorizer).serve_forever()
