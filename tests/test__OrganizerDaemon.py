from pathlib import Path

import pytest
from projectkoios.agent.organizer import (
    CategorizationProposal,
    CloudRootDiscoverer,
    FileObservation,
    LifeDomain,
    OrganizerActivity,
    OrganizerCatalog,
    OrganizerControlMode,
    OrganizerDaemon,
    ParaCategory,
)


class FixedCategorizer:
    def propose(
        self, observations: tuple[FileObservation, ...]
    ) -> tuple[CategorizationProposal, ...]:
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
    (tmp_path / "Dropbox").symlink_to(dropbox, target_is_directory=True)

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
    daemon = OrganizerDaemon(
        catalog,
        FixedCategorizer(),
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
    proposals = catalog.proposals(life_domain=LifeDomain.RESEARCH)
    assert catalog.proposal_count(LifeDomain.RESEARCH) == 1
    assert catalog.proposal_count(LifeDomain.TEACHING) == 0
    assert len(proposals) == 1
    assert proposals[0].relative_path == "research/paper.pdf"
    assert proposals[0].suggested_group == "Research references"
    assert source.read_bytes() == b"synthetic source bytes"
    assert symlink.is_symlink()
    assert any(
        value.kind == "scan_completed" for value in catalog.events_after(0)
    )


def test__organizer_catalog__rejects_symlink_database(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.sqlite3"
    target.touch()
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog_path.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        OrganizerCatalog(catalog_path)


def test__organizer_catalog__rejects_unbounded_proposal_query(
    tmp_path: Path,
) -> None:
    catalog = OrganizerCatalog(tmp_path / "catalog.sqlite3")

    try:
        catalog.proposals(limit=501)
    except ValueError as error:
        assert str(error) == "proposal limit must be between 1 and 500"
    else:
        raise AssertionError("unbounded proposal query was accepted")


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
