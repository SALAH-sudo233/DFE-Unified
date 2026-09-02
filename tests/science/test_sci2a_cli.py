import unittest

from scripts.run_sci2a_feature_interventions import denominator_fields, intervention_names


class SCI2ACLITests(unittest.TestCase):
    def test_frozen_intervention_names_and_denominators(self):
        self.assertEqual(len(intervention_names()), 8)
        self.assertEqual(denominator_fields(), ("attempts", "generated", "reconstructed", "valid", "dockable", "checked"))
