import torch
import torch.nn as nn

class MixedLoss_1d(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedLoss_1d, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='none')  # MSE for cursor position
        self.ce_loss = nn.CrossEntropyLoss()  # CE for action types

    def forward(self, outputs, labels):
        cursor_pred = outputs[:, :2]
        action_pred = outputs[:, 2:]

        cursor_true = labels[:, :2]
        action_true = labels[:, 2:]  # Assuming this is a one-hot encoded vector

        # Create a mask for cursor loss calculation, mask is 0 where cursor position is -1
        mask = (cursor_true != -1).all(dim=1).float().unsqueeze(1)  # Ensure mask dimension matches loss calculation

        # Cursor position loss calculation
        cursor_loss = self.mse_loss(cursor_pred, cursor_true)
        cursor_loss = (cursor_loss * mask).mean()  # Calculate loss only for valid cursor positions

        # Convert one-hot encoded action_true to class indices
        # torch.max returns both max values and indices along dim=1, we take indices as class labels
        _, action_true_indices = torch.max(action_true, dim=1)

        # Action type loss calculation
        action_loss = self.ce_loss(action_pred, action_true_indices)

        # Mixed loss calculation
        mixed_loss = self.alpha * cursor_loss + (1 - self.alpha) * action_loss
        return mixed_loss

class MixedLoss_nd(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedLoss_nd, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='none')  # 设置 reduction='none' 来手动控制损失的计算
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # 设置 reduction='none' 来手动控制损失的计算

    def forward(self, outputs, labels):
        num_images = outputs.shape[0]  # 获取一次传入图片的数量
        
        cursor_pred = outputs[:, :, :2].view(-1, 2)  # 将光标位置预测值重塑为 (num_images * num_samples_per_image, 2)
        action_pred = outputs[:, :, 2:].view(-1, outputs.shape[-1] - 2)  # 将动作类型预测值重塑为 (num_images * num_samples_per_image, num_actions)

        cursor_true = labels[:, :, :2].view(-1, 2)  # 将光标位置真实值重塑为 (num_images * num_samples_per_image, 2)
        action_true = labels[:, :, 2:].view(-1, labels.shape[-1] - 2)  # 将动作类型真实值重塑为 (num_images * num_samples_per_image, num_actions)

        # 创建损失计算的掩码,当光标位置为 -1 时掩码为 0,否则为 1
        mask = (cursor_true != -1).all(dim=1).float().unsqueeze(1)  # 确保掩码的维度和损失匹配

        # 计算光标位置的损失
        cursor_loss = self.mse_loss(cursor_pred, cursor_true)
        cursor_loss = (cursor_loss * mask).sum() / mask.sum()  # 只计算非 -1 位置的损失,并求平均值

        # 将独热编码的 action_true 转换为类别索引
        _, action_true_indices = torch.max(action_true, dim=1)

        # 计算动作类型的损失
        action_loss = self.ce_loss(action_pred, action_true_indices)
        action_loss = action_loss.mean()  # 求平均值

        # 混合损失
        mixed_loss = self.alpha * cursor_loss + (1 - self.alpha) * action_loss
        return mixed_loss