import torch


def vanilla_upprojection(img_feats, is_seen, point_loc, img_size=(64, 64), n_points=2048, vweights=None):
    device = img_feats.device
    b, nv, hw, c = img_feats.size(0), img_feats.size(1), img_feats.size(2), img_feats.size(3)
    img_feats = img_feats.reshape(b * nv, hw, c)
    point_loc = point_loc.reshape(b * nv, -1, 2)  # (b * nv, hw, 2)
    is_seen = is_seen.reshape(b * nv, -1, 1)  # (b * nv, hw, 1)

    # upsample to the original image size
    upsample = torch.nn.Upsample(size=img_size, mode='bilinear')  # nearest, bilinear
    avgpool = torch.nn.AvgPool2d(6, 1, 0)
    padding = torch.nn.ReplicationPad2d((2, 3, 2, 3))

    img_feats = img_feats.permute(0, 2, 1).reshape(-1, c, int(hw**0.5), int(hw**0.5))
    img_feats = avgpool(padding(img_feats))
    output = upsample(img_feats)

    # back-projecting to each points (robust to size and indices)
    n_points_eff = point_loc.size(1)
    nbatch = torch.repeat_interleave(torch.arange(0, nv * b, device=device)[:, None], n_points_eff).view(-1, ).long()
    yy = point_loc[:, :, 0].view(-1).long()
    xx = point_loc[:, :, 1].view(-1).long()

    # Clamp indices to valid image bounds
    H, W = output.size(2), output.size(3)
    yy = torch.clamp(yy, 0, H - 1)
    xx = torch.clamp(xx, 0, W - 1)

    point_feats = output[nbatch, :, yy, xx]
    point_feats = point_feats.view(b, nv, n_points_eff, -1)
    is_seen = is_seen.reshape(b, nv, n_points_eff, 1)

    # points features is the weighted mean of pixel features
    if vweights is None:
        point_feats = torch.mean(point_feats * is_seen, dim=1)
    else:
        vweights = vweights.view(1, -1, 1, 1)
        point_feats = torch.mean(point_feats * vweights * is_seen, dim=1)

    return point_feats, is_seen, point_loc