"""HKLF 5 round trips, weak-scale quantization, and exclusive outputs."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/ariadne-reduction/scripts'))
from export_hklf5 import export


def metadata():
    return {'record':'metadata','schema':'scxrd-observations/1','corrected':True,
            'all_contributors_accounted':True,'intensity_semantics':'measured_group_total',
            'phase_relation':'same_phase_supported','phase_evidence_provenance':'synthetic known same phase',
            'domain_count':2, **{key:'synthetic reference' for key in
            ('basis','backend','integration_provenance','correction_provenance','excluded_observations_provenance','overlap_model')}}


def observations():
    return [{'record':'observation','observation_id':'mixed','intensity':-12.25,'sigma':2.5,
             'contributors':[{'domain':1,'hkl':[1,2,-3]},{'domain':2,'hkl':[2,-1,3]}]},
            {'record':'observation','observation_id':'single','intensity':57.125,'sigma':1.25,
             'contributors':[{'domain':1,'hkl':[3,1,0]}]}]


def write(path, items):
    path.write_text('\n'.join(json.dumps(item) for item in [metadata(),*items])+'\n')


class ExportRound2Tests(unittest.TestCase):
    def test_normal_values_order_group_signs_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'master.jsonl'; target=root/'test.hkl'
            write(source, observations()); export(source,target)
            lines=target.read_text().splitlines()
            self.assertTrue(all(len(line)==32 for line in lines))
            self.assertEqual([int(line[28:32]) for line in lines],[-1,2,1,0])
            self.assertEqual(float(lines[0][12:20]),-12.25)
            self.assertEqual(lines[0][12:28],lines[1][12:28])
            audit=json.loads(target.with_suffix('.export.json').read_text())
            self.assertEqual(audit['intensity_and_sigma_multiplier'],1.)
            self.assertEqual(audit['observation_to_lines'],[
                {'observation_id':'mixed','first_line':1,'line_count':2},
                {'observation_id':'single','first_line':3,'line_count':1}])
            self.assertFalse(audit['format_quantization']['weak_sigma_scale_rescue'])

    def test_weak_positive_sigma_rescued_by_common_scale(self):
        items=observations()
        items[0].update(intensity=-2e-6,sigma=3e-8)
        items[1].update(intensity=4e-6,sigma=5e-8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'master.jsonl'; target=root/'test.hkl'
            write(source,items); export(source,target)
            audit=json.loads(target.with_suffix('.export.json').read_text())
            self.assertGreater(audit['intensity_and_sigma_multiplier'],1.)
            lines=target.read_text().splitlines()
            for item,line in zip(items,[lines[0],lines[2]]):
                value,sigma=float(line[12:20]),float(line[20:28])
                self.assertGreater(sigma,0)
                self.assertAlmostEqual(value/sigma,item['intensity']/item['sigma'],places=10)
            self.assertTrue(audit['format_quantization']['weak_sigma_scale_rescue'])

    def test_impossible_range_rejected_without_partial_artifacts(self):
        items=observations(); items[0].update(intensity=1e12,sigma=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'master.jsonl'; target=root/'test.hkl'
            write(source,items)
            with self.assertRaisesRegex(ValueError,'dynamic range'):
                export(source,target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.export.json').exists())

    def test_duplicate_rejected_and_existing_files_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'master.jsonl'; target=root/'test.hkl'
            items=observations(); items.append(items[0].copy()); write(source,items)
            with self.assertRaisesRegex(ValueError,'duplicated'):
                export(source,target)
            target.write_text('existing')
            with self.assertRaisesRegex(ValueError,'already exists'):
                export(source,target)
            self.assertEqual(target.read_text(),'existing')


if __name__ == '__main__':
    unittest.main()
