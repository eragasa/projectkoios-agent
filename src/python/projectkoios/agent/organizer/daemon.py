from __future__ import annotations

import time
from pathlib import Path

from projectkoios.agent.organizer.catalog import (
    CloudRootDiscoverer,
    OrganizerCatalog,
    ReadOnlyCloudCatalogIngester,
)
from projectkoios.agent.organizer.classification import FileCategorizer
from projectkoios.agent.organizer.models import (
    OrganizerActivity,
    OrganizerControlMode,
)


class OrganizerDaemon:
    """Run bounded observer scans and local categorization while enabled."""

    def __init__(
        self,
        catalog: OrganizerCatalog,
        categorizer: FileCategorizer,
        *,
        home: Path | None = None,
        classification_batch_size: int = 20,
        rescan_seconds: int = 900,
    ) -> None:
        if not 1 <= classification_batch_size <= 25:
            raise ValueError(
                "classification_batch_size must be between 1 and 25"
            )
        if not 10 <= rescan_seconds <= 86_400:
            raise ValueError("rescan_seconds is outside its bound")
        self.catalog = catalog
        self.categorizer = categorizer
        self.home = (home or Path.home()).expanduser()
        self.classification_batch_size = classification_batch_size
        self.rescan_seconds = rescan_seconds
        self.discoverer = CloudRootDiscoverer()
        self.ingester = ReadOnlyCloudCatalogIngester()

    def run_once(self) -> None:
        """Run one read-only observation and proposal pass when enabled."""
        if self.catalog.desired_mode() is not OrganizerControlMode.ON:
            self._reflect_inactive_mode()
            return
        self.catalog.set_activity(OrganizerActivity.DISCOVERING)
        roots = self.discoverer.discover(self.home)
        for root in roots:
            self.catalog.record_root(root)
        self.catalog.event("roots", f"discovered {len(roots)} cloud roots")

        observed = 0
        for root in roots:
            if self.catalog.desired_mode() is not OrganizerControlMode.ON:
                self._reflect_inactive_mode()
                return
            self.catalog.set_activity(
                OrganizerActivity.SCANNING, root_id=root.root_id
            )
            self.catalog.event(
                "scan_started",
                f"scanning {root.label}",
                root_id=root.root_id,
            )
            root_count = 0
            for observation in self.ingester.iter_files(root):
                if self.catalog.desired_mode() is not OrganizerControlMode.ON:
                    self._reflect_inactive_mode()
                    return
                self.catalog.record_file(observation)
                root_count += 1
                observed += 1
                if root_count % 500 == 0:
                    self.catalog.set_activity(
                        OrganizerActivity.SCANNING,
                        root_id=root.root_id,
                        relative_path=observation.relative_path,
                    )
                    self.catalog.event(
                        "scan_progress",
                        f"observed {root_count} files in {root.label}",
                        root_id=root.root_id,
                        file_id=observation.file_id,
                    )
            self.catalog.event(
                "scan_completed",
                f"observed {root_count} files in {root.label}",
                root_id=root.root_id,
            )
        self.catalog.event("scan_pass_completed", f"observed {observed} files")
        self._classify_pending()
        self.catalog.set_activity(OrganizerActivity.IDLE)
        self.catalog.event(
            "idle", "observer pass completed; no files were modified"
        )

    def serve_forever(self) -> None:
        """Poll control state and run without an interactive shell."""
        self.catalog.event("daemon_started", "organizer daemon started")
        while True:
            try:
                mode = self.catalog.desired_mode()
                if mode is OrganizerControlMode.ON:
                    self.run_once()
                    time.sleep(self.rescan_seconds)
                else:
                    self._reflect_inactive_mode()
                    time.sleep(1.0)
            except KeyboardInterrupt:
                self.catalog.event("daemon_stopped", "organizer daemon stopped")
                return
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self.catalog.set_activity(
                    OrganizerActivity.FAILED, error=message
                )
                self.catalog.event("error", message)
                time.sleep(5.0)

    def _classify_pending(self) -> None:
        while self.catalog.desired_mode() is OrganizerControlMode.ON:
            pending = self.catalog.pending_files(self.classification_batch_size)
            if not pending:
                return
            self.catalog.set_activity(
                OrganizerActivity.CLASSIFYING,
                root_id=pending[0].root_id,
                relative_path=pending[0].relative_path,
            )
            proposals = self.categorizer.propose(pending)
            for proposal in proposals:
                self.catalog.record_proposal(proposal)
            self.catalog.event(
                "classification_batch",
                f"proposed categories for {len(proposals)} files",
                root_id=pending[0].root_id,
                file_id=pending[0].file_id,
            )

    def _reflect_inactive_mode(self) -> None:
        mode = self.catalog.desired_mode()
        activity = (
            OrganizerActivity.PAUSED
            if mode is OrganizerControlMode.PAUSE
            else OrganizerActivity.OFF
        )
        self.catalog.set_activity(activity)
