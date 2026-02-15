from .journal_normalizer import (
    JournalNormalizer,
    get_journal_issn_l,
    get_journal_normalizer,
    is_same_journal,
    normalize_journal_name,
    refresh_journal_cache,
)
from .oa_cache import (
    OASourcesCache,
    get_oa_cache,
    is_oa_journal_cached,
    refresh_oa_cache,
)
from .open_access import (
    OAResult,
    OAStatus,
    check_oa_status,
    check_oa_status_async,
    detect_oa_from_identifiers,
    is_arxiv_id,
    is_open_access_journal,
    is_open_access_source,
)
from .Paper import Paper
from .Papers import Papers
from .Scholar import Scholar

__all__ = [
    "Paper",
    "Papers",
    "Scholar",
    "OAStatus",
    "OAResult",
    "detect_oa_from_identifiers",
    "check_oa_status",
    "check_oa_status_async",
    "is_open_access_source",
    "is_open_access_journal",
    "is_arxiv_id",
    # OA Cache
    "OASourcesCache",
    "get_oa_cache",
    "is_oa_journal_cached",
    "refresh_oa_cache",
    # Journal Normalizer
    "JournalNormalizer",
    "get_journal_normalizer",
    "normalize_journal_name",
    "get_journal_issn_l",
    "is_same_journal",
    "refresh_journal_cache",
]
