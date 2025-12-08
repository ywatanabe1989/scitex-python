#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-10 22:40:21 (ywatanabe)"


def replace(string, dict):
    for k, v in dict.items():
        string = string.replace(k, v)
    return string


# EOF
