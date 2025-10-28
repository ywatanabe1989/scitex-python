#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/git/__init__.py

"""
Git operations and utilities.
"""

from .init import (
    init_git_repo,
    find_parent_git,
    create_child_git,
    remove_child_git,
)

from .clone import (
    clone_repo,
    git_init,
)

from .commit import (
    git_add_all,
    git_commit,
)

from .branch import (
    git_branch_rename,
    git_checkout_new_branch,
)

from .remote import (
    get_remote_url,
    is_cloned_from,
)

from .workflow import (
    setup_branches,
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
]

# EOF
