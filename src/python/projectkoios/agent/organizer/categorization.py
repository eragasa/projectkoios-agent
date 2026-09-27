from __future__ import annotations

from abc import ABC, abstractmethod

from projectkoios.agent.organizer.models import (
    CategorizationProposal,
    FileObservation,
)


class BaseFileCategorizer(ABC):
    """Domain-specific contract for proposing categories from file metadata."""

    @abstractmethod
    def propose(
        self,
        observations: tuple[FileObservation, ...],
    ) -> tuple[CategorizationProposal, ...]:
        """Return one proposal for every supplied observation."""
