from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path

from projectkoios.agent.organizer.models import (
    CategorizationProposal,
    CloudRoot,
    FileAvailability,
    FileObservation,
    LifeDomain,
    OrganizerActivity,
    OrganizerControlMode,
    OrganizerEvent,
    OrganizerProposalView,
    OrganizerStatus,
    ParaCategory,
)


class CloudRootDiscoverer:
    """Discover the conventional local macOS cloud-provider roots."""

    def discover(self, home: Path | None = None) -> tuple[CloudRoot, ...]:
        resolved_home = (home or Path.home()).expanduser()
        candidates: list[tuple[str, Path, str]] = []
        cloud_storage = resolved_home / "Library" / "CloudStorage"
        if cloud_storage.is_dir() and not cloud_storage.is_symlink():
            for child in sorted(
                cloud_storage.iterdir(), key=lambda path: path.name
            ):
                if child.is_dir() and not child.is_symlink():
                    provider = self._provider(child.name)
                    candidates.append((child.name, child, provider))
        icloud = (
            resolved_home
            / "Library"
            / "Mobile Documents"
            / "com~apple~CloudDocs"
        )
        if icloud.is_dir() and not icloud.is_symlink():
            candidates.append(("iCloud Drive", icloud, "icloud"))

        roots: list[CloudRoot] = []
        seen: set[tuple[int, int]] = set()
        for label, path, provider in candidates:
            metadata = path.stat(follow_symlinks=False)
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            canonical = str(path.resolve(strict=True))
            root_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            roots.append(CloudRoot(root_id, label, path, provider))
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

    def iter_files(self, root: CloudRoot) -> Iterator[FileObservation]:
        pending = [root.path]
        while pending:
            directory = pending.pop()
            try:
                entries = tuple(os.scandir(directory))
            except OSError:
                continue
            child_directories: list[Path] = []
            for entry in sorted(entries, key=lambda item: item.name):
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                entry_path = Path(entry.path)
                if stat.S_ISDIR(metadata.st_mode):
                    child_directories.append(entry_path)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                relative_path = entry_path.relative_to(root.path).as_posix()
                file_id = hashlib.sha256(
                    f"{root.root_id}\0{relative_path}".encode()
                ).hexdigest()
                offline_flag = getattr(stat, "UF_OFFLINE", 0)
                file_flags = getattr(metadata, "st_flags", 0)
                availability = (
                    FileAvailability.CLOUD_PLACEHOLDER
                    if offline_flag and file_flags & offline_flag
                    else FileAvailability.LOCAL
                )
                yield FileObservation(
                    file_id=file_id,
                    root_id=root.root_id,
                    relative_path=relative_path,
                    name=entry.name,
                    extension=entry_path.suffix.lower(),
                    byte_size=metadata.st_size,
                    modified_ns=metadata.st_mtime_ns,
                    availability=availability,
                )
            pending.extend(reversed(child_directories))


class OrganizerCatalog:
    """Private SQLite projection for observer-mode catalog and control state."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self._prepare()

    def _prepare(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_metadata = self.path.parent.stat(follow_symlinks=False)
        if self.path.parent.is_symlink() or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            raise ValueError("organizer catalog parent must be a directory")
        if hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid():
            raise ValueError("organizer catalog parent must be user-owned")
        os.chmod(self.path.parent, 0o700)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            metadata = self.path.stat(follow_symlinks=False)
            if self.path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    "organizer catalog must be a regular file"
                ) from None
            if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
                raise ValueError(
                    "organizer catalog must be user-owned"
                ) from None
        else:
            os.close(descriptor)
        os.chmod(self.path, 0o600)
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

    def record_file(self, observation: FileObservation) -> None:
        with self._connect() as connection:
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

    def pending_files(self, limit: int) -> tuple[FileObservation, ...]:
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
            FileObservation(
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

    def proposal_count(self, life_domain: LifeDomain | None = None) -> int:
        query = "SELECT COUNT(*) FROM categorization_proposals"
        parameters: tuple[str, ...] = ()
        if life_domain is not None:
            query += " WHERE life_domain = ?"
            parameters = (life_domain.value,)
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise RuntimeError("organizer proposal count is unavailable")
        return int(row[0])

    def proposals(
        self,
        *,
        life_domain: LifeDomain | None = None,
        limit: int = 200,
    ) -> tuple[OrganizerProposalView, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("proposal limit must be between 1 and 500")
        condition = ""
        parameters: tuple[object, ...] = (limit,)
        if life_domain is not None:
            condition = "WHERE p.life_domain = ?"
            parameters = (life_domain.value, limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    f.file_id, f.root_id, f.relative_path, f.name,
                    f.extension, f.byte_size, f.availability,
                    p.para_category, p.life_domain, p.confidence,
                    p.suggested_group, p.rationale, p.model,
                    p.model_digest, p.proposed_at
                FROM categorization_proposals AS p
                JOIN file_observations AS f ON f.file_id = p.file_id
                {condition}
                ORDER BY p.confidence DESC, f.root_id, f.relative_path
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return tuple(
            OrganizerProposalView(
                file_id=str(row["file_id"]),
                root_id=str(row["root_id"]),
                relative_path=str(row["relative_path"]),
                name=str(row["name"]),
                extension=str(row["extension"]),
                byte_size=int(row["byte_size"]),
                availability=FileAvailability(str(row["availability"])),
                para_category=ParaCategory(str(row["para_category"])),
                life_domain=LifeDomain(str(row["life_domain"])),
                confidence=float(row["confidence"]),
                suggested_group=str(row["suggested_group"]),
                rationale=str(row["rationale"]),
                model=str(row["model"]),
                model_digest=str(row["model_digest"]),
                proposed_at=str(row["proposed_at"]),
            )
            for row in rows
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
