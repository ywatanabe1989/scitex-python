#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/__init__.py

"""
Git operations and utilities.
"""

from ._init import (
    init_git_repo,
    find_parent_git,
    create_child_git,
    remove_child_git,
)

from ._clone import (
    clone_repo,
    git_init,
)

from ._commit import (
    git_add_all,
    git_commit,
)

from ._branch import (
    git_branch_rename,
    git_checkout_new_branch,
)

from ._remote import (
    get_remote_url,
    is_cloned_from,
)

from ._workflow import (
    setup_branches,
)

from ._retry import (
    git_retry,
)

__all__ = [
    "init_git_repo",
    "find_parent_git",
    "create_child_git",
    "remove_child_git",
    "clone_repo",
    "git_init",
    "git_add_all",
    "git_commit",
    "git_branch_rename",
    "git_checkout_new_branch",
    "get_remote_url",
    "is_cloned_from",
    "setup_branches",
    "git_retry",
]

# EOF
