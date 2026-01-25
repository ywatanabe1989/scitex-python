#!/usr/bin/env python3
# Time-stamp: "2024-11-02 13:43:36 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/os/_check_host.py

import socket
import sys


def check_host(keyword):
    """Check if the current hostname contains the given keyword."""
    return keyword in socket.gethostname()


is_host = check_host


def verify_host(keyword):
    if is_host(keyword):
        print(f"Host verification successed for keyword: {keyword}")
        return
    else:
        print(f"Host verification failed for keyword: {keyword}")
        sys.exit(1)


if __name__ == "__main__":
    # check_host("ywata")
    verify_host("titan")
    verify_host("ywata")
    verify_host("crest")


# EOF
