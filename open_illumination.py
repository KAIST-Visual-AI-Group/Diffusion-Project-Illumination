import json
from urllib.request import urlopen
import huggingface_hub
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", default="./illumination_data")
parser.add_argument("--light", required=True, choices=['lighting_patterns', 'OLAT'])
parser.add_argument("--obj_id", default=1, type=int)
parser.add_argument("--with_raw", default=False, action='store_true')

args = parser.parse_args()


def main():
    data_json = json.loads(urlopen("https://oppo-us-research.github.io/OpenIllumination/data.json").read())
    data_olat = json.loads(urlopen("https://oppo-us-research.github.io/OpenIllumination/data_olat.json").read())
    repo_id = "OpenIllumination/OpenIllumination"
    data = data_json if args.light == 'lighting_patterns' else data_olat
    
    for obj_id in range(len(data['obj_list'])):
        obj = data['obj_list'][obj_id]

        save_root_path = os.path.join(args.local_dir, args.light, obj['data_name'])
        save_output_path = os.path.join(save_root_path, 'output')
        save_lights_path = os.path.join(save_root_path, 'Lights')
        print(save_lights_path, save_output_path)
        if os.path.exists(save_output_path) and os.path.exists(save_lights_path):
            if len(os.listdir(save_output_path)) == 5 and len(os.listdir(save_lights_path)) == 13:
                print(f"Already downloaded {obj['data_name']}. Skipping...")
                continue

        allow_patterns = [f"*{args.light}/{obj['data_name']}/Lights/*/raw_undistorted/*",
                        f"*{args.light}/{obj['data_name']}/output/*"
                        ]
        if args.light != "lighting_patterns" and args.with_raw:
            allow_patterns.append(f"OLAT/{obj['data_name']}/RAW/*")
        huggingface_hub.snapshot_download(repo_id,
                                        repo_type="dataset",
                                        allow_patterns=allow_patterns,
                                        local_dir=args.local_dir)


if __name__ == '__main__':
    main()
