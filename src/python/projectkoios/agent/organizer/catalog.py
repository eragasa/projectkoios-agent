from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path

from projectkoios.agent.organizer.models import (
    CatalogFileObservation,
    CategorizationProposal,
    CloudRoot,
    FileAvailability,
    OrganizerActivity,
    OrganizerControlMode,
    OrganizerEvent,
    OrganizerStatus,
)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_MACOS_SF_DATALESS = 0x40000000
_PLACEHOLDER_FLAGS = _MACOS_SF_DATALESS | getattr(stat, "UF_OFFLINE", 0)


class CloudRootDiscoverer:
    """Discover the conventional local macOS cloud-provider roots."""

    def discover(self, home: Path | None = None) -> tuple[CloudRoot, ...]:
        absolute_home = (home or Path.home()).expanduser().absolute()
        candidates: list[tuple[str, Path, str, tuple[int, int]]] = []
        cloud_storage = absolute_home / "Library" / "CloudStorage"
        try:
            cloud_descriptor = _open_directory_path(cloud_storage)
        except OSError:
            cloud_descriptor = None
        if cloud_descriptor is not None:
            try:
                try:
                    with os.scandir(cloud_descriptor) as scanner:
                        entries = tuple(
                            sorted(scanner, key=lambda item: item.name)
                        )
                except OSError:
                    entries = ()
                for entry in entries:
                    try:
                        child_descriptor = os.open(
                            entry.name,
                            _DIRECTORY_OPEN_FLAGS,
                            dir_fd=cloud_descriptor,
                        )
                    except OSError:
                        continue
                    try:
                        try:
                            metadata = os.fstat(child_descriptor)
                        except OSError:
                            continue
                        if not stat.S_ISDIR(metadata.st_mode):
                            continue
                        candidates.append(
                            (
                                entry.name,
                                cloud_storage / entry.name,
                                self._provider(entry.name),
                                (metadata.st_dev, metadata.st_ino),
                            )
                        )
                    finally:
                        os.close(child_descriptor)
            finally:
                os.close(cloud_descriptor)

        icloud = (
            absolute_home
            / "Library"
            / "Mobile Documents"
            / "com~apple~CloudDocs"
        )
        try:
            icloud_descriptor = _open_directory_path(icloud)
        except OSError:
            icloud_descriptor = None
        if icloud_descriptor is not None:
            try:
                try:
                    icloud_metadata = os.fstat(icloud_descriptor)
                except OSError:
                    icloud_metadata = None
                if icloud_metadata is not None:
                    candidates.append(
                        (
                            "iCloud Drive",
                            icloud,
                            "icloud",
                            (
                                icloud_metadata.st_dev,
                                icloud_metadata.st_ino,
                            ),
                        )
                    )
            finally:
                os.close(icloud_descriptor)

        roots: list[CloudRoot] = []
        seen: set[tuple[int, int]] = set()
        for label, path, provider, identity in candidates:
            if identity in seen:
                continue
            root_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
            try:
                root = CloudRoot(root_id, label, path, provider)
            except ValueError:
                continue
            seen.add(identity)
            roots.append(root)
        return tuple(roots)

    @staticmethod
    def _provider(name: str) -> str:
        lowered = name.lower()
        if lowered.startswith("dropbox"):
            return "dropbox"
        if lowered.startswith("googledrive"):
            return "google_drive"
        return "cloud_storage"


class ReadOnlyCloudCatalogIngester:
    """Yield metadata observations without reading file payload bytes."""

    def iter_files(self, root: CloudRoot) -> Iterator[CatalogFileObservation]:
        try:
            root_descriptor = _open_directory_path(root.path)
        except OSError:
            return
        try:
            yield from self._iter_directory(root, root_descriptor, "")
        finally:
            os.close(root_descriptor)

    def _iter_directory(
        self,
        root: CloudRoot,
        directory_descriptor: int,
        relative_parent: str,
    ) -> Iterator[CatalogFileObservation]:
        try:
            with os.scandir(directory_descriptor) as scanner:
                entries = tuple(sorted(scanner, key=lambda item: item.name))
        except OSError:
            return

        child_directories: list[tuple[str, str]] = []
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                continue
            relative_path = (
                f"{relative_parent}/{entry.name}"
                if relative_parent
                else entry.name
            )
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append((entry.name, relative_path))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            file_id = hashlib.sha256(
                f"{root.root_id}\0{relative_path}".encode()
            ).hexdigest()
            try:
                observation = CatalogFileObservation(
                    file_id=file_id,
                    root_id=root.root_id,
                    relative_path=relative_path,
                    name=entry.name,
                    extension=Path(entry.name).suffix.lower(),
                    byte_size=metadata.st_size,
                    modified_ns=metadata.st_mtime_ns,
                    availability=_availability_from_flags(
                        int(getattr(metadata, "st_flags", 0))
                    ),
                )
            except ValueError:
                continue
            yield observation

        for child_name, relative_path in child_directories:
            try:
                child_descriptor = os.open(
                    child_name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_descriptor,
                )
            except OSError:
                continue
            try:
                yield from self._iter_directory(
                    root,
                    child_descriptor,
                    relative_path,
                )
            finally:
                os.close(child_descriptor)


def _open_directory_path(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("directory path must be absolute")
    descriptor = os.open(path.anchor, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            child_descriptor = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _availability_from_flags(flags: int) -> FileAvailability:
    if flags & _PLACEHOLDER_FLAGS:
        return FileAvailability.CLOUD_PLACEHOLDER
    return FileAvailability.LOCAL


class OrganizerCatalog:
    """Private SQLite projection for observer-mode catalog and control state."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        if not self.path.is_absolute():
            raise ValueError("organizer catalog path must be absolute")
        self._prepare()

    def _prepare(self) -> None:
        if self.path.is_symlink():
            raise ValueError("organizer catalog path must not be a symlink")
        if self.path.parent.is_symlink():
            raise ValueError(
                "organizer catalog directory must not be a symlink"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.parent.is_symlink():
            raise ValueError(
                "organizer catalog directory must not be a symlink"
            )
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS organizer_control (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    desired_mode TEXT NOT NULL,
                    activity TEXT NOT NULL,
                    current_root_id TEXT,
                    current_relative_path TEXT,
                    last_error TEXT
                );
                INSERT OR IGNORE INTO organizer_control
                    (singleton, desired_mode, activity)
                    VALUES (1, 'off', 'off');
                CREATE TABLE IF NOT EXISTS cloud_roots (
                    root_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    path TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS file_observations (
                    file_id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL REFERENCES cloud_roots(root_id),
                    relative_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    availability TEXT NOT NULL,
                    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(root_id, relative_path)
                );
                CREATE TABLE IF NOT EXISTS categorization_proposals (
                    file_id TEXT PRIMARY KEY
                        REFERENCES file_observations(file_id),
                    para_category TEXT NOT NULL,
                    life_domain TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    suggested_group TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_digest TEXT NOT NULL,
                    proposed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS organizer_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    root_id TEXT,
                    file_id TEXT
                );
                """
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def set_mode(self, mode: OrganizerControlMode) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE organizer_control
                SET desired_mode = ?
                WHERE singleton = 1
                """,
                (mode.value,),
            )
            connection.execute(
                "INSERT INTO organizer_events(kind, message) VALUES (?, ?)",
                ("control", f"organizer mode requested: {mode.value}"),
            )

    def desired_mode(self) -> OrganizerControlMode:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT desired_mode FROM organizer_control WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("organizer control row is unavailable")
        return OrganizerControlMode(str(row["desired_mode"]))

    def set_activity(
        self,
        activity: OrganizerActivity,
        *,
        root_id: str | None = None,
        relative_path: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE organizer_control
                SET activity = ?, current_root_id = ?,
                    current_relative_path = ?, last_error = ?
                WHERE singleton = 1
                """,
                (activity.value, root_id, relative_path, error),
            )

    def record_root(self, root: CloudRoot) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cloud_roots(root_id, label, path, provider)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(root_id) DO UPDATE SET
                    label = excluded.label,
                    path = excluded.path,
                    provider = excluded.provider,
                    observed_at = CURRENT_TIMESTAMP
                """,
                (root.root_id, root.label, str(root.path), root.provider),
            )

    def record_file(self, observation: CatalogFileObservation) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM categorization_proposals
                WHERE file_id = ? AND EXISTS (
                    SELECT 1 FROM file_observations
                    WHERE file_id = ? AND (
                        name != ? OR extension != ? OR byte_size != ?
                        OR modified_ns != ? OR availability != ?
                    )
                )
                """,
                (
                    observation.file_id,
                    observation.file_id,
                    observation.name,
                    observation.extension,
                    observation.byte_size,
                    observation.modified_ns,
                    observation.availability.value,
                ),
            )
            connection.execute(
                """
                INSERT INTO file_observations(
                    file_id, root_id, relative_path, name, extension,
                    byte_size, modified_ns, availability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    name = excluded.name,
                    extension = excluded.extension,
                    byte_size = excluded.byte_size,
                    modified_ns = excluded.modified_ns,
                    availability = excluded.availability,
                    observed_at = CURRENT_TIMESTAMP
                """,
                (
                    observation.file_id,
                    observation.root_id,
                    observation.relative_path,
                    observation.name,
                    observation.extension,
                    observation.byte_size,
                    observation.modified_ns,
                    observation.availability.value,
                ),
            )

    def pending_files(self, limit: int) -> tuple[CatalogFileObservation, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.* FROM file_observations AS f
                LEFT JOIN categorization_proposals AS p ON p.file_id = f.file_id
                WHERE p.file_id IS NULL
                ORDER BY f.root_id, f.relative_path
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            CatalogFileObservation(
                file_id=str(row["file_id"]),
                root_id=str(row["root_id"]),
                relative_path=str(row["relative_path"]),
                name=str(row["name"]),
                extension=str(row["extension"]),
                byte_size=int(row["byte_size"]),
                modified_ns=int(row["modified_ns"]),
                availability=FileAvailability(str(row["availability"])),
            )
            for row in rows
        )

    def record_proposal(self, proposal: CategorizationProposal) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO categorization_proposals(
                    file_id, para_category, life_domain, confidence,
                    suggested_group, rationale, model, model_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    para_category = excluded.para_category,
                    life_domain = excluded.life_domain,
                    confidence = excluded.confidence,
                    suggested_group = excluded.suggested_group,
                    rationale = excluded.rationale,
                    model = excluded.model,
                    model_digest = excluded.model_digest,
                    proposed_at = CURRENT_TIMESTAMP
                """,
                (
                    proposal.file_id,
                    proposal.para_category.value,
                    proposal.life_domain.value,
                    proposal.confidence,
                    proposal.suggested_group,
                    proposal.rationale,
                    proposal.model,
                    proposal.model_digest,
                ),
            )

    def event(
        self,
        kind: str,
        message: str,
        *,
        root_id: str | None = None,
        file_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO organizer_events(kind, message, root_id, file_id)
                VALUES (?, ?, ?, ?)
                """,
                (kind, message, root_id, file_id),
            )

    def events_after(
        self, sequence: int, limit: int = 200
    ) -> tuple[OrganizerEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, occurred_at, kind, message, root_id, file_id
                FROM organizer_events
                WHERE sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (sequence, limit),
            ).fetchall()
        return tuple(
            OrganizerEvent(
                sequence=int(row["sequence"]),
                occurred_at=str(row["occurred_at"]),
                kind=str(row["kind"]),
                message=str(row["message"]),
                root_id=None if row["root_id"] is None else str(row["root_id"]),
                file_id=None if row["file_id"] is None else str(row["file_id"]),
            )
            for row in rows
        )

    def status(self) -> OrganizerStatus:
        with self._connect() as connection:
            control = connection.execute(
                "SELECT * FROM organizer_control WHERE singleton = 1"
            ).fetchone()
            counts = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM cloud_roots) AS roots,
                    (SELECT COUNT(*) FROM file_observations) AS files,
                    (SELECT COUNT(*) FROM file_observations
                        WHERE availability = 'local') AS local_files,
                    (SELECT COUNT(*) FROM file_observations
                        WHERE availability = 'cloud_placeholder')
                        AS placeholders,
                    (SELECT COUNT(*) FROM categorization_proposals)
                        AS proposals,
                    (SELECT COALESCE(MAX(sequence), 0)
                        FROM organizer_events) AS events
                """
            ).fetchone()
        if control is None or counts is None:
            raise RuntimeError("organizer catalog status is unavailable")
        return OrganizerStatus(
            desired_mode=OrganizerControlMode(str(control["desired_mode"])),
            activity=OrganizerActivity(str(control["activity"])),
            discovered_roots=int(counts["roots"]),
            observed_files=int(counts["files"]),
            local_files=int(counts["local_files"]),
            placeholder_files=int(counts["placeholders"]),
            proposed_files=int(counts["proposals"]),
            last_event_sequence=int(counts["events"]),
            current_root_id=(
                None
                if control["current_root_id"] is None
                else str(control["current_root_id"])
            ),
            current_relative_path=(
                None
                if control["current_relative_path"] is None
                else str(control["current_relative_path"])
            ),
            last_error=(
                None
                if control["last_error"] is None
                else str(control["last_error"])
            ),
        )
