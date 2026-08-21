# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Any


def require_cutlass_module_hash(
    dsl: Any,
    module: Any,
    module_hash: str | None,
    function_name: str,
) -> str:
    """Return a valid key before converting a no-cache compile into a cached one."""

    if module_hash is None:
        get_module_hash = getattr(dsl, "get_module_hash", None)
        if not callable(get_module_hash):
            raise RuntimeError(
                "CUTLASS disk caching requires BaseDSL.get_module_hash when "
                "the caller supplies module_hash=None"
            )
        module_hash = get_module_hash(module, function_name)
    if not isinstance(module_hash, str) or not module_hash:
        raise RuntimeError("CUTLASS module hash must be a non-empty string")
    return module_hash
