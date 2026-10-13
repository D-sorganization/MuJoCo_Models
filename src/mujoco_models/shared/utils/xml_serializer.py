# SPDX-License-Identifier: MIT
"""XML serialization helpers for MuJoCo MJCF ElementTrees."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable

logger = logging.getLogger(__name__)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add whitespace indentation to an ElementTree in-place.

    Delegates to :func:`xml.etree.ElementTree.indent` (stdlib, Python 3.9+).
    The *level* parameter is accepted for backward compatibility but ignored;
    the stdlib implementation always starts from the root indentation level.
    """
    ET.indent(elem, space="  ", level=level)
    if level == 0:
        elem.tail = "\n"


def _fast_serialize_node(  # noqa: C901
    elem: ET.Element,
    buffer_append: Callable[[str], None],
    level: int = 0,
) -> None:
    """Recursively serialize an ElementTree node into a string buffer.

    This avoids the significant overhead of xml.etree.ElementTree.tostring()
    while preserving necessary XML escaping for correctness.
    """
    tag = elem.tag

    # ⚡ Bolt Optimization:
    # Combine formatting strings using f-strings into a single append call.
    buffer_append(f"<{tag}")

    attrib = elem.attrib
    if attrib:
        for k, v in attrib.items():
            if "&" in v or "<" in v or '"' in v or "\n" in v or "\r" in v or "\t" in v:
                v = (
                    v.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace('"', "&quot;")
                    .replace("\n", "&#10;")
                    .replace("\r", "&#13;")
                    .replace("\t", "&#9;")
                )
            # ⚡ Bolt Optimization:
            # Combine formatting strings using f-strings into a single append call.
            # This is significantly faster than calling buffer_append 5 times.
            buffer_append(f' {k}="{v}"')

    has_children = bool(len(elem))

    text = elem.text
    has_text = text is not None and not text.isspace()

    if not has_children and not has_text:
        buffer_append(" />")
    else:
        buffer_append(">")

        if has_text and text is not None:
            if "&" in text or "<" in text:
                text = text.replace("&", "&amp;").replace("<", "&lt;")
            buffer_append(text)

        if has_children:
            buffer_append("\n")
            next_level = level + 1
            indent_next = "  " * next_level
            for child in elem:
                buffer_append(indent_next)
                _fast_serialize_node(child, buffer_append, next_level)
                buffer_append("\n")
            buffer_append("  " * level)

        # ⚡ Bolt Optimization:
        # Reduce buffer_append calls for closing tags
        buffer_append(f"</{tag}>")

    tail = elem.tail
    if tail is not None and not tail.isspace():
        if "&" in tail or "<" in tail:
            tail = tail.replace("&", "&amp;").replace("<", "&lt;")
        buffer_append(tail)


def serialize_model(root: ET.Element) -> str:
    """Serialize a MuJoCo MJCF ElementTree to a formatted XML string."""
    logger.debug("Serializing MJCF model with root tag=%s", root.tag)
    # ⚡ Bolt Optimization:
    # Use custom recursive serialization instead of ET.tostring for speed.
    # Avoiding ET.indent() saves a full O(N) tree traversal pass just to add whitespace.
    # We pass buffer.append to avoid attribute lookup overhead
    # in the recursive calls, and inline the fast-path string checks and escaping
    # avoiding the overhead of list.extend with tuple creation and Python call frames.
    buf: list[str] = ["<?xml version='1.0' encoding='utf-8'?>\n"]
    _fast_serialize_node(root, buf.append, 0)
    buf.append("\n")
    return "".join(buf)
