"""Mass observations and solver-specific cloth settings."""

from math import isclose, isfinite

REFERENCE_BENDING_STIFFNESS = 1.45


def mass_observation(record):
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("Garment mass must record kg and the measurement source")
    kg, source = record.get("kg"), record.get("source")
    if type(kg) not in (int, float) or not isfinite(kg) or kg <= 0:
        raise ValueError("Garment mass must be a positive finite mass in kg")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Identify the independent source of the garment mass")
    return {"kg": float(kg), "source": source}


def material_mass(material):
    return mass_observation(material.get("garment_mass")) if isinstance(material, dict) else None


def verify_mass_settings(observation, settings):
    if not isinstance(settings, dict):
        raise ValueError("Simulation did not record the supplied garment mass")
    count, per_vertex = settings.get("mesh_vertex_count"), settings.get("vertex_mass_kg")
    if (settings.get("requested_garment_mass_kg") != observation["kg"]
            or settings.get("distribution") != "uniform_over_sewn_vertices"
            or type(count) is not int or count <= 0 or type(per_vertex) not in (int, float)
            or not isclose(per_vertex * count, observation["kg"], rel_tol=1e-6)):
        raise ValueError("Simulation vertex masses do not sum to the recorded garment mass")


def blender_bending(record):
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("Record the Blender bending stiffness and its source")
    stiffness, source = record.get("stiffness"), record.get("source")
    if type(stiffness) not in (int, float) or not isfinite(stiffness) or stiffness <= 0:
        raise ValueError("Blender bending stiffness must be positive and finite")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Identify the calibration source for the Blender bending stiffness")
    return {"stiffness": float(stiffness), "source": source}


def material_bending(material):
    return blender_bending(material.get("blender_bending")) if isinstance(material, dict) else None


def verify_bending_settings(observation, settings):
    used = settings.get("stiffness") if isinstance(settings, dict) else None
    if (type(used) not in (int, float) or not isclose(used, observation["stiffness"], rel_tol=1e-6)
            or settings.get("model") != "ANGULAR"):
        raise ValueError("Simulation does not match the recorded Blender angular bending stiffness")
