from pathlib import Path

import pytest
from projectkoios.agent.organizer import (
    BaseFileCategorizer,
    CategorizationProposal,
    CloudRoot,
    CloudRootDiscoverer,
    FileAvailability,
    FileObservation,
    LifeDomain,
    OrganizerActivity,
    OrganizerCatalog,
    OrganizerControlMode,
    OrganizerDaemon,
    ParaCategory,
    ReadOnlyCloudCatalogIngester,
)

# isort: split
from projectkoios.agent.organizer import catalog as organizer_catalog


class FixedCategorizer(BaseFileCategorizer):
    def __init__(self) -> None:
        self.call_count = 0

    def propose(
        self, observations: tuple[FileObservation, ...]
    ) -> tuple[CategorizationProposal, ...]:
        self.call_count += 1
        return tuple(
            CategorizationProposal(
                file_id=value.file_id,
                para_category=ParaCategory.RESOURCE,
                life_domain=LifeDomain.RESEARCH,
                confidence=0.8,
                suggested_group="Research references",
                rationale="The path and extension indicate research material.",
                model="fixture",
                model_digest="0" * 64,
            )
            for value in observations
        )


def test__cloud_root_discoverer__finds_provider_roots_without_symlink_duplicate(
    tmp_path: Path,
) -> None:
    cloud_storage = tmp_path / "Library" / "CloudStorage"
    dropbox = cloud_storage / "Dropbox"
    google = cloud_storage / "GoogleDrive-example"
    icloud = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    dropbox.mkdir(parents=True)
    google.mkdir()
    icloud.mkdir(parents=True)
    (cloud_storage / "DropboxAlias").symlink_to(
        dropbox, target_is_directory=True
    )

    roots = CloudRootDiscoverer().discover(tmp_path)

    assert tuple(value.label for value in roots) == (
        "Dropbox",
        "GoogleDrive-example",
        "iCloud Drive",
    )
    assert tuple(value.provider for value in roots) == (
        "dropbox",
        "google_drive",
        "icloud",
    )


def test__cloud_root_discoverer__rejects_replaced_provider_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud_storage = tmp_path / "Library" / "CloudStorage"
    dropbox = cloud_storage / "Dropbox"
    outside = tmp_path / "outside"
    dropbox.mkdir(parents=True)
    outside.mkdir()
    original_open = organizer_catalog.os.open
    replaced = False

    def replace_before_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if path == "Dropbox" and dir_fd is not None and not replaced:
            dropbox.rmdir()
            dropbox.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(organizer_catalog.os, "open", replace_before_open)

    roots = CloudRootDiscoverer().discover(tmp_path)

    assert replaced
    assert roots == ()


def test__cloud_file_flags__recognize_macos_dataless_placeholders() -> None:
    assert (
        organizer_catalog._availability_from_flags(0x40000000)
        is FileAvailability.CLOUD_PLACEHOLDER
    )
    assert (
        organizer_catalog._availability_from_flags(0) is FileAvailability.LOCAL
    )


def test__cloud_ingester__does_not_follow_replaced_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "root"
    child = root_path / "child"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()
    (outside / "private.txt").write_text("private", encoding="utf-8")
    root = CloudRoot("a" * 64, "fixture", root_path, "fixture")
    original_open = organizer_catalog.os.open
    replaced = False

    def replace_before_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if path == "child" and dir_fd is not None:
            child.rmdir()
            child.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(organizer_catalog.os, "open", replace_before_open)

    observations = tuple(ReadOnlyCloudCatalogIngester().iter_files(root))

    assert replaced
    assert observations == ()


def test__cloud_ingester__skips_metadata_outside_domain_bounds(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / f"unsupported.{('x' * 33)}").touch()
    (root_path / "valid.txt").touch()
    root = CloudRoot("a" * 64, "fixture", root_path, "fixture")

    observations = tuple(ReadOnlyCloudCatalogIngester().iter_files(root))

    assert tuple(item.relative_path for item in observations) == ("valid.txt",)


def test__organizer_daemon__catalogs_and_classifies_without_mutating_sources(
    tmp_path: Path,
) -> None:
    dropbox = tmp_path / "Library" / "CloudStorage" / "Dropbox"
    nested = dropbox / "research"
    nested.mkdir(parents=True)
    source = nested / "paper.pdf"
    source.write_bytes(b"synthetic source bytes")
    symlink = nested / "paper-link.pdf"
    symlink.symlink_to(source)
    catalog = OrganizerCatalog(tmp_path / "state" / "catalog.sqlite3")
    catalog.set_mode(OrganizerControlMode.ON)
    categorizer = FixedCategorizer()
    daemon = OrganizerDaemon(
        catalog,
        categorizer,
        home=tmp_path,
        classification_batch_size=5,
        rescan_seconds=10,
    )

    daemon.run_once()

    status = catalog.status()
    assert status.desired_mode is OrganizerControlMode.ON
    assert status.activity is OrganizerActivity.IDLE
    assert status.discovered_roots == 1
    assert status.observed_files == 1
    assert status.proposed_files == 1
    assert source.read_bytes() == b"synthetic source bytes"
    assert symlink.is_symlink()
    assert categorizer.call_count == 1
    assert any(
        value.kind == "scan_completed" for value in catalog.events_after(0)
    )

    source.write_bytes(b"changed synthetic source bytes")
    daemon.run_once()

    assert source.read_bytes() == b"changed synthetic source bytes"
    assert categorizer.call_count == 2
    assert catalog.status().proposed_files == 1


def test__organizer_catalog__rejects_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        OrganizerCatalog(Path("catalog.sqlite3"))

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "catalog-link"
    symlink.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="directory must not be a symlink"):
        OrganizerCatalog(symlink / "catalog.sqlite3")


def test__organizer_catalog__pause_and_off_are_explicit_control_states(
    tmp_path: Path,
) -> None:
    catalog = OrganizerCatalog(tmp_path / "catalog.sqlite3")

    catalog.set_mode(OrganizerControlMode.PAUSE)
    OrganizerDaemon(
        catalog,
        FixedCategorizer(),
        home=tmp_path,
        rescan_seconds=10,
    ).run_once()
    paused = catalog.status()
    catalog.set_mode(OrganizerControlMode.OFF)
    OrganizerDaemon(
        catalog,
        FixedCategorizer(),
        home=tmp_path,
        rescan_seconds=10,
    ).run_once()
    off = catalog.status()

    assert paused.activity is OrganizerActivity.PAUSED
    assert off.activity is OrganizerActivity.OFF
