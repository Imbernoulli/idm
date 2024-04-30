class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        初始化混合损失函数
        :param alpha: 用于平衡光标位置损失和动作类型损失的系数
        """
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()  # 用于光标位置
        self.ce_loss = nn.CrossEntropyLoss()  # 用于动作类型

    def forward(self, outputs, labels):
        """
        计算损失函数
        :param outputs: 模型输出，假设为 (N, 2 + num_actions) 的形式，
                        其中前2个是光标位置，剩余是动作类型的 logits
        :param labels: 真实标签，光标位置 (N, 2) 和动作类型 (N,)
        """
        # 分割输出为光标位置和动作类型
        cursor_pred = outputs[:, :2]
        action_pred = outputs[:, 2:]

        # 分割标签为光标位置和动作类型
        cursor_true = labels[:, :2]
        action_true = labels[:, 2].long()  # 确保动作类型标签是长整型

        # 计算两个损失
        cursor_loss = self.mse_loss(cursor_pred, cursor_true)
        action_loss = self.ce_loss(action_pred, action_true)

        # 混合损失
        mixed_loss = self.alpha * cursor_loss + (1 - self.alpha) * action_loss
        return mixed_loss