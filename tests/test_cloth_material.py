from unittest import TestCase

from superarchiveallclo.cloth_material import mass_observation, material_mass, verify_mass_settings


class ClothMaterialTest(TestCase):
    def test_optional_mass_requires_value_and_provenance(self):
        self.assertIsNone(material_mass({"fiber_percent": {"cotton": 100}}))
        for kg in (False, float("inf"), float("nan"), 0, -1, "0.255"):
            with self.subTest(kg=kg), self.assertRaisesRegex(ValueError, "positive finite mass"):
                mass_observation({"kg": kg, "source": "scale"})
        with self.assertRaisesRegex(ValueError, "independent source"):
            mass_observation({"kg": 0.255, "source": "  "})

    def test_reported_per_vertex_value_must_close_the_mass_budget(self):
        observation = material_mass({"garment_mass": {"kg": 0.255, "source": "scale"}})
        settings = {"mesh_vertex_count": 6848, "vertex_mass_kg": 3.7237150536384434e-05,
                    "requested_garment_mass_kg": 0.255, "distribution": "uniform_over_sewn_vertices"}
        verify_mass_settings(observation, settings)
        with self.assertRaisesRegex(ValueError, "do not sum"):
            verify_mass_settings(observation, {**settings, "vertex_mass_kg": 0.15})
