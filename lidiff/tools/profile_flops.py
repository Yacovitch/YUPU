"""Profile FLOPs for the YUPU CLIP+normal diffusion model.

PyTorch's profiler counts supported dense operators (for example matrix
multiplication and dense convolution). MinkowskiEngine CUDA kernels are custom
operators and may report zero FLOPs; this script surfaces those uncounted
kernels instead of silently presenting the result as complete.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

import lidiff.datasets.datasets as datasets
from lidiff.models.models_clip_normal import DiffusionPoints


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '-ckpt', required=True, help='trained Lightning .ckpt file')
    parser.add_argument(
        '--config', '-c',
        default=str(Path(__file__).resolve().parents[1] / 'config/yupu_normal_config.yaml'),
        help='experiment YAML; validation/test always use real sparse inputs',
    )
    parser.add_argument('--output', '-o', default='flops_report.json', help='JSON report path')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--sample-index', type=int, default=0)
    return parser.parse_args()


def move_batch_to_cuda(batch):
    return {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def profile_forward(label, callback):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            callback()
        torch.cuda.synchronize()

    events = prof.key_averages(group_by_input_shape=True)
    counted_flops = int(sum(int(event.flops or 0) for event in events))
    uncounted_cuda = sorted(
        (
            {
                'operator': event.key,
                'self_cuda_time_us': float(event.self_cuda_time_total),
            }
            for event in events
            if event.self_cuda_time_total > 0 and not event.flops
        ),
        key=lambda item: item['self_cuda_time_us'],
        reverse=True,
    )[:25]
    return {
        'label': label,
        'counted_flops': counted_flops,
        'counted_gflops': counted_flops / 1.0e9,
        'uncounted_cuda_operators_by_time': uncounted_cuda,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('FLOPs profiling requires CUDA because the model and MinkowskiEngine are CUDA-only')

    with open(args.config, 'r') as config_file:
        cfg = yaml.safe_load(config_file)

    # Profiling must use real sparse inputs, even when the supplied config is
    # the synthetic-training experiment.
    cfg['data']['synthetic_downsample_train'] = False
    cfg['train']['batch_size'] = 1
    cfg['train']['num_workers'] = 0

    model = DiffusionPoints.load_from_checkpoint(
        args.checkpoint,
        hparams=cfg,
        config_path=args.config,
    ).cuda().eval()

    data_module = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    loader = data_module.val_dataloader() if args.split == 'validation' else data_module.test_dataloader()
    dataset = loader.dataset
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f'sample-index {args.sample_index} is outside dataset of size {len(dataset)}')
    batch = loader.collate_fn([dataset[args.sample_index]])
    batch = move_batch_to_cuda(batch)

    x_full = model.points_to_tensor(batch['pcd_full'], batch['mean'], batch['std'])
    x_full_sparse = x_full.sparse()
    x_part = model.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
    x_normals = model.points_to_tensor_with_features(
        batch['pcd_part'], batch['normals'], batch['mean'], batch['std'], like=x_part
    )
    zeros = torch.zeros_like(batch['pcd_part'])
    zero_stats = torch.zeros_like(batch['mean'])
    x_uncond = model.points_to_tensor(zeros, zero_stats, torch.zeros_like(batch['std']))
    timestep = torch.zeros(batch['pcd_full'].shape[0], dtype=torch.long, device='cuda')

    conditional = lambda: model.forward(
        x_full, x_full_sparse, x_part, x_normals, timestep,
        batch['pcd_part'], batch['mean'], batch['std'],
    )
    unconditional = lambda: model.forward(
        x_full, x_full_sparse, x_uncond, x_normals, timestep,
        zeros, zero_stats, torch.zeros_like(batch['std']),
    )

    # Warm up lazy CUDA kernels and coordinate maps before measuring.
    with torch.no_grad():
        conditional()
        unconditional()
    torch.cuda.synchronize()

    conditional_report = profile_forward('conditional_denoising_forward', conditional)
    unconditional_report = profile_forward('unconditional_denoising_forward', unconditional)
    guided_step_flops = conditional_report['counted_flops'] + unconditional_report['counted_flops']

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'config': str(Path(args.config).resolve()),
        'split': args.split,
        'sample_index': args.sample_index,
        'input_points': int(batch['pcd_part'].shape[1]),
        'output_points': int(batch['pcd_full'].shape[1]),
        'denoising_steps': int(cfg['diff']['s_steps']),
        'parameters': {'total': total_params, 'trainable': trainable_params},
        'conditional_forward': conditional_report,
        'unconditional_forward': unconditional_report,
        'classifier_free_guided_step': {
            'counted_flops': guided_step_flops,
            'counted_gflops': guided_step_flops / 1.0e9,
        },
        'full_sampling_trajectory_estimate': {
            'counted_flops': guided_step_flops * int(cfg['diff']['s_steps']),
            'counted_gflops': guided_step_flops * int(cfg['diff']['s_steps']) / 1.0e9,
        },
        'coverage_note': (
            'Counted FLOPs include only operators supported by torch.profiler. '
            'MinkowskiEngine and other custom CUDA kernels listed with zero FLOPs are not included; '
            'therefore the reported total is a lower bound.'
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Wrote FLOPs report to {output_path}')


if __name__ == '__main__':
    main()
