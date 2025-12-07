from .Paper import Paper
from .Papers import Papers
from .Scholar import Scholar
from .open_access import (
    OAStatus,
    OAResult,
    detect_oa_from_identifiers,
    check_oa_status,
    check_oa_status_async,
    is_open_access_source,
    is_open_access_journal,
    is_arxiv_id,
)
from .oa_cache import (
    OASourcesCache,
    get_oa_cache,
    is_oa_journal_cached,
    refresh_oa_cache,
)
from .journal_normalizer import (
    JournalNormalizer,
    get_journal_normalizer,
    normalize_journal_name,
    get_journal_issn_l,
    is_same_journal,
    refresh_journal_cache,
)

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
