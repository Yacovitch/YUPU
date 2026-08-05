import os
import numpy as np
import open3d as o3d
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall, CompletionIoU, RMSE, HausdorffDistance, EarthMoversDistance 
import tqdm
from natsort import natsorted
from lidiff.tools.diff_completion_pipeline import DiffCompletion
from lidiff.utils.histogram_metrics import compute_hist_metrics 
import click
import json

PATH_DATA = '/nas2/jacob/data/YUPU_data_bin/dataset/sequences/L_T2_619947_4847977'
PATH_DATA_GT = '/nas2/jacob/data/YUPU_data_bin/dataset/sequences_gt/L_T2_619947_4847977'

completion_iou = CompletionIoU()
rmse = RMSE()
chamfer_distance = ChamferDistance()
precision_recall = PrecisionRecall(0.50,2*0.50,100)
hausdorff_distance = HausdorffDistance()
earth_movers_distance = EarthMoversDistance()


def get_scan_completion(scan_path, path, diff_completion, max_range, save_reverse_diffusion):
    pcd_file = os.path.join(PATH_DATA, 'velodyne', scan_path)
    points = np.fromfile(pcd_file, dtype=np.float32)
    points = points.reshape(-1,4) 
    dist = np.sqrt(np.sum(points[:,:3]**2, axis=-1))
    input_points = points[dist < max_range, :3]
    if diff_completion is None:
        pred_path = f'{scan_path.split(".")[0]}.ply' #f'{scan_path.split(".")[0]}_refine.ply' #for refined result.
        #print(pred_path)
        pcd_pred = o3d.io.read_point_cloud(os.path.join(path, pred_path))
        points = np.array(pcd_pred.points)
        dist = np.sqrt(np.sum(points**2, axis=-1))
        pcd_pred.points = o3d.utility.Vector3dVector(points[dist < max_range])
    else:
        complete_scan_refine, complete_scan, scan = diff_completion.complete_scan(points, save_reverse_diffusion)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(complete_scan_refine)

    return pcd_pred, input_points



@click.command()
@click.option('--path', '-p', type=str, default='/nas2/jacob/LiDiff/lidiff/results/yupu_low_res_frozen_clip_cross_fusion/diff', help='path to the scan sequence')
@click.option('--voxel_size', '-v', type=float, default=0.05, help='voxel size')
@click.option('--max_range', '-m', type=float, default=50, help='max range')
@click.option('--denoising_steps', '-t', type=int, default=50, help='number of denoising steps')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weights')
@click.option('--diff', '-d', type=str, default='', help='run diffusion pipeline')
@click.option('--refine', '-r', type=str, default='/nas2/jacob/LiDiff/lidiff/checkpoints/refine_net.ckpt', help='path to the checkpoint for refinement net')
@click.option('--save_reverse_diffusion', '-s', type=bool, default=False, help='save reverse diffusion')
def main(path, voxel_size, max_range, denoising_steps, cond_weight, diff, refine, save_reverse_diffusion): 
    diff_completion = None

    jsd_3d = []
    jsd_bev = []

    for scan_path in tqdm.tqdm(list(natsorted(os.listdir(f'{PATH_DATA}/velodyne')))):
        pcd_pred, cur_scan = get_scan_completion(scan_path, path, diff_completion, max_range, save_reverse_diffusion)
        
        pcd_file = os.path.join(PATH_DATA_GT, 'velodyne', scan_path)
        point_gt = np.fromfile(pcd_file, dtype=np.float32).reshape(-1,4)
        point_gt = point_gt[:,:3]
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(point_gt)

        jsd_3d.append(compute_hist_metrics(pcd_gt, pcd_pred, bev=False))
        jsd_bev.append(compute_hist_metrics(pcd_gt, pcd_pred, bev=True))
        #print(f'JSD 3D: {jsd_3d[-1]}')
        #print(f'JSD BEV: {jsd_bev[-1]}')

        rmse.update(pcd_gt, pcd_pred)
        completion_iou.update(pcd_gt, pcd_pred)
        chamfer_distance.update(pcd_gt, pcd_pred)
        precision_recall.update(pcd_gt, pcd_pred)
        hausdorff_distance.update(pcd_gt, pcd_pred)
        earth_movers_distance.update(pcd_gt, pcd_pred)
        
        rmse_mean, rmse_std = rmse.compute()
        #print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
        thr_ious = completion_iou.compute()
        #for v_size in thr_ious.keys():
        #    print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
        cd_mean, cd_std = chamfer_distance.compute()
        #print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        pr, re, f1 = precision_recall.compute_auc()
        #print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')
        hd_mean, hd_std = hausdorff_distance.compute()
        #print(f'HD Mean: {hd_mean}\tHD Std: {hd_std}')
        emd_mean, emd_std = earth_movers_distance.compute()
        #print(f'EMD Mean: {emd_mean}\tEMD Std: {emd_std}')



    print('\n\n=================== FINAL RESULTS ===================\n\n')
    print(f'JSD 3D: {np.array(jsd_3d).mean()}')
    print(f'JSD BEV: {np.array(jsd_bev).mean()}')
    print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
    thr_ious = completion_iou.compute()
    for v_size in thr_ious.keys():
        print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
    cd_mean, cd_std = chamfer_distance.compute()
    print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
    pr, re, f1 = precision_recall.compute_auc()
    print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')
    hd_mean, hd_std = hausdorff_distance.compute()
    print(f'HD Mean: {hd_mean}\tHD Std: {hd_std}')
    emd_mean, emd_std = earth_movers_distance.compute()
    print(f'EMD Mean: {emd_mean}\tEMD Std: {emd_std}')
    
    res_dict = {
        'jsd': np.array(jsd_bev).mean(),
        'jsd_noclip_3d': np.array(jsd_3d).mean(),
        'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'ious': thr_ious,
        'cd_mean': cd_mean, 'cd_std': cd_std,
        'pr': pr, 're': re, 'f1': f1,
        'hd_mean': hd_mean, 'hd_std': hd_std,
        'emd_mean': emd_mean, 'emd_std': emd_std,
    }

    log_path = os.path.join(*path.split('/')[:-1])
    with open(f'/{log_path}/res_log.yaml', 'w+') as log_res:
        json.dump(res_dict, log_res)

if __name__ == '__main__':
    main()
