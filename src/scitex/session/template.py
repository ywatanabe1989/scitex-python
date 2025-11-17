#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-18 09:08:38 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/session/template.py


from pprint import pprint
import scitex as stx


@stx.session(verbose=False)
def main(
    arg1=None,
    arg2=None,
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Demonstration for scitex.session.session"""
    pprint(CONFIG)


if __name__ == "__main__":
    main()

# EOF
