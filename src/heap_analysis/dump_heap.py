"""Script to be injected with either Pyrasite or sys.remote_exec to dump the heap of a running Python process to a file.

The output is a JSONL file where each line is a JSON object representing an object in the heap,
with its id, type, referers, references, and optionally its value (for simple types)."""

from types import ModuleType, FunctionType

import gc
import json
import sys


def _dump_heap():
    # Get all objects tracked by gc
    all_objects = gc.get_objects()
    print(f"Found {len(all_objects)} objects from gc.get_objects()", file=sys.stderr)
    extra_objects = []
    object_ids_tracked = set(id(obj) for obj in all_objects)
    _value_types = (str, bytes, int, float, complex, bool, type(None))

    # Collect ids of our own data structures so we can exclude them
    _exclude_ids = set(
        [id(all_objects), id(extra_objects), id(object_ids_tracked), id(_value_types)]
    )

    # Max length for string representations of values
    _max_value_len = 1000

    def _maybe_add_ref(ref_obj, refs):
        ref_id = id(ref_obj)
        if ref_id not in _exclude_ids:
            refs.append(ref_id)
            if ref_id not in object_ids_tracked:
                extra_objects.append(ref_obj)
                object_ids_tracked.add(ref_id)

    def _get_all_references(obj):
        """Get references from an object, including non-gc-tracked immutables."""
        refs = []
        refs_set_id = id(refs)
        _exclude_ids.add(refs_set_id)

        obj_type = type(obj)
        if obj_type is dict:
            for k, v in obj.items():
                _maybe_add_ref(k, refs)
                _maybe_add_ref(v, refs)
        elif obj_type in (list, tuple, frozenset, set):
            for item in obj:
                _maybe_add_ref(item, refs)
        else:
            # Fall back to gc.get_referents for other types,
            # which catches __dict__, slots, etc.
            gc_refs = gc.get_referents(obj)
            _exclude_ids.add(id(gc_refs))
            for r in gc_refs:
                _maybe_add_ref(r, refs)

        return refs

    def _get_qualname(obj):
        """Get a qualified name for an object, if possible."""
        if qualname := getattr(obj, "__qualname__", None):
            return qualname
        elif name := getattr(obj, "__name__", None):
            return name
        else:
            return repr(obj)

    def _get_type_name(obj):
        """Get a friendly type name for an object."""
        t = type(obj)
        module = getattr(t, "__module__", None)
        qualname = _get_qualname(t)
        if module and module != "builtins":
            return f"{module}.{qualname}"
        return qualname

    _SENTINEL = object()
    _exclude_ids.add(id(_SENTINEL))

    def _string_extractor(obj):
        if len(obj) > _max_value_len:
            return obj[:_max_value_len] + "...<truncated>"
        return obj

    def _bytes_extractor(obj):
        if len(obj) > _max_value_len:
            return repr(obj[:_max_value_len]) + "...<truncated>"
        return repr(obj)

    def _module_extractor(obj):
        return f"module {obj.__name__}"

    def _function_extractor(obj):
        return f"{_get_qualname(obj)} (function in module {obj.__module__})"

    def _staticmethod_extractor(obj):
        return f"{_get_qualname(obj)} (staticmethod in module {obj.__module__})"

    def _classmethod_extractor(obj):
        return f"{_get_qualname(obj)} (classmethod in module {obj.__module__})"

    def _type_extractor(obj):
        return f"{_get_qualname(obj)} (type in module {obj.__module__})"

    _value_extractors = {
        str: _string_extractor,
        bytes: _bytes_extractor,
        int: lambda x: x,
        float: lambda x: x,
        complex: str,
        bool: lambda x: x,
        type(None): lambda x: None,
        ModuleType: _module_extractor,
        FunctionType: _function_extractor,
        staticmethod: _staticmethod_extractor,
        classmethod: _classmethod_extractor,
        type: _type_extractor,
    }

    try:
        with open("/tmp/dump.jsonl", "w") as f:
            _exclude_ids.add(id(f))

            def dump_object(obj):
                obj_id = id(obj)

                # Skip our own bookkeeping objects
                if obj_id in _exclude_ids:
                    return

                # Get references (including non-gc-tracked children)
                references = _get_all_references(obj)

                type_name = _get_type_name(obj)

                record = {
                    "id": obj_id,
                    "type": type_name,
                    "references": references,
                    "size": sys.getsizeof(obj, 0),  # Get size of object, excluding referents
                    # Don't get referrers - it's too slow. We'll index it offline later.
                }

                # Only include value for whitelisted types
                if extractor := _value_extractors.get(type(obj)):
                    try:
                        record["value"] = extractor(obj)
                    except Exception as e:
                        record["value"] = f"<error extracting value: {e}>"

                _exclude_ids.add(id(record))
                _exclude_ids.add(id(record.get("references")))

                line = json.dumps(record, default=str)
                _exclude_ids.add(id(line))
                f.write(line)
                f.write("\n")

            for obj in all_objects:
                dump_object(obj)
            for obj in extra_objects:
                dump_object(obj)

    except Exception as e:
        sys.stderr.write(f"dump_heap error: {e}\n")


_dump_heap()
