"""Export a single-element Abaqus ODB to JSON with Abaqus Python 2.7.

Usage: abaqus python export_abaqus_single_element.py input.odb input.inp output.json
"""
from __future__ import print_function

import json
import sys

from odbAccess import openOdb


def plastic_table(inp_path):
    """Read the true yield stress / equivalent plastic strain table."""
    table = []
    reading = False
    with open(inp_path) as stream:
        for line in stream:
            if line.lower().startswith('*plastic'):
                reading = True
                continue
            if reading and line.startswith('*'):
                break
            if reading and line.strip():
                stress, strain = [float(x) for x in line.split(',')[:2]]
                table.append([strain, stress])
    return table


def export(odb_path, inp_path, output_path):
    odb = openOdb(odb_path, readOnly=True)
    try:
        step = odb.steps['Step-1']
        instance = odb.rootAssembly.instances['PART-2-1']
        result = {
            'source': {'odb': odb_path, 'inp': inp_path},
            'hardening': plastic_table(inp_path),
            'nodes': {str(node.label): [float(x) for x in node.coordinates[:2]] for node in instance.nodes},
            'elements': {str(element.label): [int(x) for x in element.connectivity] for element in instance.elements},
            'frames': [],
        }
        for frame in step.frames:
            fields = {}
            for name in frame.fieldOutputs.keys():
                output = frame.fieldOutputs[name]
                fields[name] = {
                    'components': list(output.componentLabels),
                    'values': [
                        {'node': int(value.nodeLabel) if getattr(value, 'nodeLabel', None) is not None else None,
                         'element': int(value.elementLabel) if getattr(value, 'elementLabel', None) is not None else None,
                         'data': [float(x) for x in value.data] if hasattr(value.data, '__len__') else float(value.data)}
                        for value in output.values
                    ],
                }
            result['frames'].append({'time': float(frame.frameValue), 'fields': fields})
        result['history'] = {
            name: [[float(time), float(value)] for time, value in output.data]
            for region in step.historyRegions.values()
            for name, output in region.historyOutputs.items()
        }
        with open(output_path, 'w') as stream:
            json.dump(result, stream, sort_keys=True, separators=(',', ':'))
    finally:
        odb.close()


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], sys.argv[3])
