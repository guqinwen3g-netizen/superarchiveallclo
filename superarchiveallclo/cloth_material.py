"""Measured garment mass and its uniform discretization for the cloth solver."""

from math import isclose, isfinite


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
