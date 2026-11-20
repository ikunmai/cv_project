"""Run one complete crystal video, preserving official tracking/filter settings."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default='ma2035_023')
    parser.add_argument('--variant', choices=['quality', 'baseline'], default='quality')
    parser.add_argument('--quality_min_confidence', type=float, default=0.30)
    parser.add_argument('--quality_min_side', type=float, default=2.0)
    parser.add_argument('--quality_nms_iou', type=float, default=0.80)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.quality_min_confidence <= 1 or not 0 <= args.quality_nms_iou <= 1 or args.quality_min_side < 0:
        parser.error('Invalid quality filtering thresholds')
    root = Path(__file__).resolve().parent
    import os
    os.chdir(root)
    data = Path('VideoCount/Crystals')
    gt = json.loads((data/'anno/crystals-count-gt.json').read_text())
    if args.video not in gt:
        parser.error(f'Unknown video: {args.video}')
    video_dir = data/'frames'/args.video
    script = 'count_in_videos_pseudo_exemplars_quality.py' if args.variant == 'quality' else 'count_in_videos_pseudo_exemplars.py'
    for f in [video_dir, Path('checkpoints/countgd_box.pth'), Path('checkpoints/sam2.1_hiera_large.pt')]:
        if not f.exists():
            raise FileNotFoundError(f)
    # Unique directory: repeated runs never truncate previous predictions or the baseline.
    out = Path('results') / (f'crystals_{args.variant}_{args.video}_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    predicted = out/'predicted.json'
    commands = []
    for text in gt[args.video]:
        cmd = [sys.executable, '-u', script, '--video_dir', str(video_dir), '--input_text', text,
               '--sam_checkpoint', 'checkpoints/sam2.1_hiera_large.pt',
               '--sam_model_cfg', 'configs/sam2.1/sam2.1_hiera_l.yaml',
               '--obj_batch_size', '30', '--img_batch_size', '1', '--downsample_factor', '1',
               '--pretrain_model_path', 'checkpoints/countgd_box.pth',
               '--output_file', str(predicted), '--temp_dir', str(out/'temp_frames'),
               '--temporal_filter', '--w', '3', '--convert_to_rgb']
        if args.variant == 'quality':
            cmd += ['--quality_min_confidence', str(args.quality_min_confidence),
                    '--quality_min_side', str(args.quality_min_side),
                    '--quality_nms_iou', str(args.quality_nms_iou)]
        commands.append(cmd)
    print('Video:', args.video, 'Variant:', args.variant, 'Results:', out, flush=True)
    if args.dry_run:
        import shlex
        for cmd in commands:
            print(shlex.join(cmd))
        return
    out.mkdir(parents=True, exist_ok=False)
    predicted.write_text('{}')
    (out/'ground-truth.json').write_text(json.dumps({args.video: gt[args.video]}, indent=2))
    files = [script, 'pseudo_exemplar_quality.py', Path(__file__).name]
    metadata = {'settings': vars(args), 'commands': commands,
                'source_sha256': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    (out/'settings.json').write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    try:
        with (out/'run.log').open('w', buffering=1) as log:
            for cmd in commands:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                    for line in child.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    code = child.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, cmd)
        result = json.loads(predicted.read_text())
        assert set(result[str(video_dir)]) == set(gt[args.video]), 'Incomplete predictions'
        evaluation = subprocess.run([sys.executable, 'evaluate_counting_accuracy.py',
                        '--ground_truth', str(out/'ground-truth.json'), '--predicted', str(predicted),
                        '--parent_dir', str(data/'frames')], capture_output=True, text=True, check=True)
        print(evaluation.stdout, flush=True)
        (out/'metrics.txt').write_text(evaluation.stdout)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'incomplete'
        metadata['error'] = repr(error)
        raise
    finally:
        metadata['elapsed_seconds'] = time.monotonic()-start
        (out/'settings.json').write_text(json.dumps(metadata, indent=2))
        print('Results saved to:', out, flush=True)


if __name__ == '__main__':
    main()
