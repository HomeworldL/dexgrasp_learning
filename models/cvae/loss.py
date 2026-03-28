import torch


def cvae_loss(hand_gt, hand_pred, mean, log_var):
    loss_trans = (hand_pred[:, :3] - hand_gt[:, :3]).abs().mean()
    loss_rot = (hand_pred[:, 3:12] - hand_gt[:, 3:12]).abs().mean()
    loss_kld = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum()
    return {
        "loss_trans": loss_trans,
        "loss_rot": loss_rot,
        "loss_kld": loss_kld,
        "loss": loss_trans + loss_rot + loss_kld,
    }
