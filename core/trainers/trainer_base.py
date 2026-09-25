import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class SupervisedTrainer:
    def __init__(self, model, optimizer, device, logger, out_dir,
                 epochs=20, early_stop=5, criterion=None, use_amp=False):
        """
        model       : 网络模型
        optimizer   : 优化器
        device      : GPU/CPU
        logger      : 日志对象
        out_dir     : 模型保存目录
        epochs      : 训练轮数
        early_stop  : 提前停止轮数
        criterion   : 损失函数 (默认 CrossEntropy)
        use_amp     : 是否启用混合精度训练 (AMP)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.out_dir = out_dir
        self.epochs = epochs
        self.early_stop = early_stop
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.use_amp = use_amp

        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.log("⚡ 启用混合精度训练 (AMP) 模式")

    # ==============================================================
    def _one_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, total_samples = 0.0, 0
        all_preds, all_labels = [], []

        for x, y in tqdm(loader, desc="train" if train else "val", ncols=100):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.detach().cpu().numpy())

        avg_loss = total_loss / max(1, total_samples)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, acc

    # ==============================================================
    def fit(self, train_loader, val_loader):
        best_acc, no_improve = 0.0, 0
        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._one_epoch(train_loader, train=True)
            val_loss, val_acc = self._one_epoch(val_loader, train=False)

            self.logger.log(f"[Epoch {epoch:02d}] TrainLoss={tr_loss:.4f} Acc={tr_acc:.4f} | "
                            f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                torch.save(self.model.state_dict(), f"{self.out_dir}/best.pt")
                self.logger.log(f"✅ 模型更新 (ValAcc={val_acc:.4f}) → 已保存 best.pt")
            else:
                no_improve += 1
                if no_improve >= self.early_stop:
                    self.logger.log("⏹️ 提前停止训练 (Validation 无提升)")
                    break

        self.logger.log(f"🏁 训练完成，最佳验证准确率 = {best_acc:.4f}")
