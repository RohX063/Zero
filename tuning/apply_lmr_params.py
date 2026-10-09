#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('state', help='SPSA state JSON')
ap.add_argument('output', help='Output JSON parameter file')
args = ap.parse_args()
state = json.loads(Path(args.state).read_text())
params = state['params']
out = {
    'LMRBase': {'value': params['LMRBase']},
    'LMRDepthCoeff': {'value': params['LMRDepthCoeff']},
    'LMRMoveCoeff': {'value': params['LMRMoveCoeff']},
    'interaction_fixed': 503.4375,
    'units': '1/1024 ply'
}
Path(args.output).write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
