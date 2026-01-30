import json
import os

import numpy as np
import streamlit as st
import torch

SAVE_FILE = "dungeon_tensor_save.json"


def serialize_value(val):
    """Recursive serialization for nested lists/dicts and special types."""
    if isinstance(val, torch.Tensor):
        return {
            "__tensor__": True,
            "data": val.tolist(),
            "dtype": str(val.dtype).replace("torch.", ""),
            "device": "cpu",  # Always save as CPU
        }
    elif isinstance(val, (set, tuple)):
        # Convert set/tuple to list, mark the type
        return {
            "__type__": type(val).__name__,
            "data": [serialize_value(item) for item in val],
        }
    elif isinstance(val, list):
        return [serialize_value(item) for item in val]
    elif isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (np.integer, np.floating)):
        return val.item()  # Convert numpy scalars to python scalars

    return val


def deserialize_value(val):
    """Recursive deserialization."""
    if isinstance(val, dict):
        if val.get("__tensor__"):
            dtype_str = val["dtype"]
            # specific safe attribute lookup
            try:
                dtype = getattr(torch, dtype_str)
            except AttributeError:
                dtype = torch.float32  # fallback

            # create tensor
            try:
                return torch.tensor(val["data"], dtype=dtype)
            except Exception as e:
                print(
                    f"Warning: Failed to create tensor with dtype {dtype_str}, falling back to default. Error: {e}"
                )
                return torch.tensor(val["data"])  # fallback without dtype

        if val.get("__type__") == "set":
            return set(deserialize_value(item) for item in val["data"])
        elif val.get("__type__") == "tuple":
            return tuple(deserialize_value(item) for item in val["data"])

        # Regular dict recursion
        return {k: deserialize_value(v) for k, v in val.items()}

    elif isinstance(val, list):
        return [deserialize_value(item) for item in val]

    return val


def save_game():
    """Saves the current st.session_state to a JSON file."""
    state_to_save = {}

    # Filter out Streamlit internal keys if necessary,
    # generally they start with specialized prefixes or are not standard session_state items set by user
    # We iterate strict keys we care about?
    # Or just iterate all and try-catch.
    # Iterating all is safer to capture dynamic keys like 'saved_frq_...'

    for key, val in st.session_state.items():
        # Skip UI component keys that shouldn't be persisted (like button states or temp inputs)
        # Often temp inputs have generic generated keys, but user-named keys are fine.
        # Let's save everything that is serializable.
        try:
            # We don't check json.dumps here because we have custom serializer
            # But we might want to skip keys that are clearly widgets (FormSubmitter?)
            if isinstance(key, str) and (
                key.startswith("FormSubmitter")
                or key.startswith("btn_")
                or key == "authentication_status"
            ):
                continue

            serialized = serialize_value(val)
            # Verify json serializability
            # json.dumps(serialized) # Optional check, adds overhead but strictness
            state_to_save[key] = serialized
        except Exception as e:
            print(f"Skipping key '{key}' during save due to serialization error: {e}")

    try:
        with open(SAVE_FILE, "w") as f:
            json.dump(state_to_save, f, indent=2)
        st.toast("Game Saved Successfully!", icon="💾")
    except Exception as e:
        st.error(f"Failed to save game: {e}")


def load_game():
    """Loads game state from file."""
    if not os.path.exists(SAVE_FILE):
        st.error("No save file found!")
        return False

    try:
        with open(SAVE_FILE, "r") as f:
            saved_state = json.load(f)

        # Clear current state or update?
        # Update is safer to keep default keys if not present in save?
        # But if we want to "Reload", we probably want to wipe current state and replace with saved.
        # But `init_game` might have set defaults.
        # Let's use st.session_state.update

        for key, val in saved_state.items():
            if key.startswith("btn_"):
                continue
            st.session_state[key] = deserialize_value(val)

        st.toast("Game Loaded Successfully!", icon="📂")
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Failed to load game: {e}")
        return False
