import copy

# import math
import os
import time
from typing import Optional

import numpy as np
import torch

from tsfb.base.origin_code.tgcrn.utils.logger import get_logger
from tsfb.base.origin_code.tgcrn.utils.metrics import All_Metrics

# import sys


class Trainer(object):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        args,
        graph_loader,
        lr_scheduler=None,
        ct_factor=0.3,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        # self.time_loss = torch.nn.MSELoss().to(args.device)
        self.optimizer = optimizer
        self.train_loader = train_loader

        # self.train_num_batch = train_loader.num_batch
        self.train_num_batch = len(train_loader)

        self.val_loader = val_loader
        self.test_loader = test_loader
        self.graph_loader = graph_loader
        self.scaler = scaler
        self.a = ct_factor
        self.args = args
        self.lr_scheduler = lr_scheduler
        if val_loader is not None:
            # self.val_per_epoch = val_loader.num_batch
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.log_dir, "best_model.pth")
        self.loss_figure_path = os.path.join(self.args.log_dir, "loss.png")
        # log
        if os.path.isdir(args.log_dir) is False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info("Experiment log path in: {}".format(args.log_dir))
        self.logger.info(args)

    def _prepare_input(self, inputs):
        """Prepare and processing input data."""
        if self.args.time_embedding and self.args.od_flag:
            X, X_TIME, Y, Y_TIME, OD, DO = inputs
            X_TIME = X_TIME.to(self.args.device)
            Y_TIME = Y_TIME.to(self.args.device)
            OD = OD.to(self.args.device)
            DO = DO.to(self.args.device)
            return X, X_TIME, Y, Y_TIME, OD, DO
        elif self.args.time_embedding and self.args.constrative_time:
            X, X_TIME, Y, Y_TIME, Cons_time = inputs
            X_TIME = X_TIME.to(self.args.device)
            Y_TIME = Y_TIME.to(self.args.device)
            Cons_time = Cons_time.to(self.args.device)
            return X, X_TIME, Y, Y_TIME, Cons_time
        elif self.args.time_embedding:
            X, X_TIME, Y, Y_TIME = inputs
            X_TIME = X_TIME.to(self.args.device)
            Y_TIME = Y_TIME.to(self.args.device)
            return X, X_TIME, Y, Y_TIME
        else:
            X, Y = inputs
            return X, Y

    def _get_model_output(
        self,
        X: torch.Tensor,
        X_TIME: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        Y_TIME: Optional[torch.Tensor] = None,
        OD: Optional[torch.Tensor] = None,
        DO: Optional[torch.Tensor] = None,
        Cons_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the model’s forward pass.

        Args:
          X       – [B, seq_len, N, F] input features
          X_TIME  – [B, seq_len]  input slice indices
          Y       – [B, horizon, N, F]  (unused here)
          Y_TIME  – [B, horizon] target slice indices
          OD, DO  – optional OD adjacency
          Cons_time – optional contrastive‐time samples
        """
        # 1) Move to device & squeeze-away any spurious last dim
        X = X.to(self.args.device)
        if X_TIME is not None:
            X_TIME = X_TIME.squeeze(-1).to(self.args.device)
        if Y_TIME is not None:
            Y_TIME = Y_TIME.squeeze(-1).to(self.args.device)
        if OD is not None:
            OD = OD.to(self.args.device)
            DO = DO.to(self.args.device)
        if Cons_time is not None:
            Cons_time = Cons_time.to(self.args.device)

        # 2) Build the single-tuple that TGCRN.forward expects
        if self.args.time_embedding and self.args.node_mode == "kgc":
            # dynamic KG‐sampling branch
            self.graph_loader._genA_()
            adj_ent = self.graph_loader.adj_ent.to(self.args.device)
            adj_rel = self.graph_loader.adj_rel.to(self.args.device)
            inputs = (X, X_TIME, Y_TIME, adj_ent, adj_rel)

        elif self.args.time_embedding and self.args.constrative_time:
            # contrastive‐time branch
            inputs = (X, X_TIME, Y_TIME, Cons_time)

        elif self.args.time_embedding:
            # regular time‐embedding (with or without OD)
            if self.args.od_flag:
                inputs = (X, X_TIME, Y_TIME, OD, DO)
            else:
                inputs = (X, X_TIME, Y_TIME)

        else:
            # plain‐X (no time embeddings)
            inputs = (X,)

        # 3) Forward in one shot
        return self.model(inputs)

    def _calculate_loss(self, output, Y, ratios=None):
        """Calculate loss."""
        if self.args.real_value:
            output = self.scaler.inverse_transform(output)
            Y = self.scaler.inverse_transform(Y)

        loss = self.loss(output, Y)

        if self.args.constrative_time and ratios is not None:
            time_loss = (
                self.loss(ratios[0], ratios[1])
                + self.loss(ratios[0], ratios[2])
                + self.loss(ratios[2], ratios[1])
            ) / 3.0
            loss += self.a * time_loss

        return loss

    def train_epoch(self, epoch):
        """Training one epoch."""
        self.model.train()
        total_loss = 0

        for iter, inputs in enumerate(self.train_loader):
            # prepare input
            input_data = self._prepare_input(inputs)
            if isinstance(input_data, tuple):
                X = input_data[0]
                X = X[..., : self.args.input_dim].to(self.args.device)
                Y = input_data[2][..., : self.args.output_dim].to(self.args.device)
            else:
                X = input_data[..., : self.args.input_dim].to(self.args.device)
                Y = inputs[1][..., : self.args.output_dim].to(self.args.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.args.constrative_time:
                output, ratios = self._get_model_output(*input_data)
                loss = self._calculate_loss(output, Y, ratios)
            else:
                output = self._get_model_output(*input_data)
                loss = self._calculate_loss(output, Y)

            # Backward pass
            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            self.optimizer.step()
            total_loss += loss.item()

            # Log
            if iter % self.args.log_step == 0:
                self.logger.info(
                    f"Train Epoch {epoch}: {iter}/{self.train_num_batch} "
                    f"Loss: {loss.item():.6f}"
                )

        train_epoch_loss = total_loss / self.train_num_batch
        self.logger.info(
            f"**********Train Epoch {epoch}: averaged Loss: {train_epoch_loss:.6f}"
        )

        if self.args.lr_decay:
            self.lr_scheduler.step()

        return train_epoch_loss

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for iter, inputs in enumerate(val_dataloader):
                if self.args.time_embedding and self.args.od_flag:
                    X, X_TIME, Y, Y_TIME, OD, DO = inputs
                    X_TIME = X_TIME.to(self.args.device)
                    Y_TIME = Y_TIME.to(self.args.device)
                    OD = OD.to(self.args.device)
                    DO = DO.to(self.args.device)
                elif self.args.time_embedding:
                    X, X_TIME, Y, Y_TIME = inputs
                    X_TIME = X_TIME.to(self.args.device)
                    Y_TIME = Y_TIME.to(self.args.device)
                else:
                    X, Y = inputs
                X = X[..., : self.args.input_dim].to(self.args.device)
                Y = Y[..., : self.args.output_dim].to(self.args.device)

                if self.args.time_embedding and self.args.node_mode == "kgc":
                    adj_ent, adj_rel = self.graph_loader.adj_ent.to(
                        X.device
                    ), self.graph_loader.adj_rel.to(X.device)
                    output = self.model((X, X_TIME, Y_TIME, adj_ent, adj_rel))  # 保持数据格式
                elif self.args.time_embedding:
                    if self.args.od_flag:
                        output = self.model((X, X_TIME, Y_TIME, OD, DO))
                    else:
                        output = self.model((X, X_TIME, Y_TIME))
                else:
                    output = self.model(X)

                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    Y = self.scaler.inverse_transform(Y)
                loss = self.loss(output, Y)

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info(
            "**********Val Epoch {}: average Loss: {:.6f}".format(epoch, val_loss)
        )
        return val_loss

    def train(self):
        best_model = None
        best_loss = float("inf")
        patient = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            # 当val_loader为空，全部数据用于训练
            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                patient = 0
                best_state = True
            else:
                patient += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if patient == self.args.early_stop_patience:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.args.early_stop_patience)
                    )
                    break
            # save the best state
            if best_state is True:
                self.logger.info(
                    "*********************************Current best model saved!"
                )
                # 采用变量copy的方式，减少磁盘读写时间．
                best_model = copy.deepcopy(self.model.state_dict())
            # self.test(
            #   self.model,
            #   self.args,
            #   self.test_loader,
            #   self.scaler,
            #   self.logger,
            #   self.graph_loader
            # )

        training_time = time.time() - start_time
        self.logger.info(
            "Total training time: {:.4f}min, best loss: {:.6f}".format(
                (training_time / 60), best_loss
            )
        )

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(
            self.model,
            self.args,
            self.test_loader,
            self.scaler,
            self.logger,
            graph_loader=self.graph_loader,
        )

        # self.test(self.model, self.args, self.test_loader,
        #           self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.args,
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None, graph_loader=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point["state_dict"]
            args = check_point["config"]
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for iter, inputs in enumerate(data_loader):
                if args.time_embedding and args.od_flag:
                    X, X_TIME, Y, Y_TIME, OD, DO = inputs
                    X_TIME = X_TIME.to(args.device)
                    Y_TIME = Y_TIME.to(args.device)
                    OD = OD.to(args.device)
                    DO = DO.to(args.device)
                elif args.time_embedding:
                    X, X_TIME, Y, Y_TIME = inputs
                    X_TIME = X_TIME.to(args.device)
                    Y_TIME = Y_TIME.to(args.device)
                else:
                    X, Y = inputs

                X = X[..., : args.input_dim].to(args.device)
                Y = Y[..., : args.output_dim].to(args.device)

                if args.time_embedding and args.node_mode == "kgc":
                    adj_ent, adj_rel = graph_loader.adj_ent.to(
                        X.device
                    ), graph_loader.adj_rel.to(X.device)
                    output = model((X, X_TIME, Y_TIME, adj_ent, adj_rel))  # 保持数据格式
                elif args.time_embedding:
                    if args.od_flag:
                        output = model((X, X_TIME, Y_TIME, OD, DO))
                    else:
                        output = model((X, X_TIME, Y_TIME))
                else:
                    output = model(X)

                y_true.append(Y)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        else:
            y_pred = torch.cat(y_pred, dim=0)

        y_true, y_pred = y_true[: args.test_len, ...], y_pred[: args.test_len, ...]
        np.save(os.path.join(args.log_dir, "true.npy"), y_true.cpu().numpy())
        np.save(os.path.join(args.log_dir, "pred.npy"), y_pred.cpu().numpy())

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(
                y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh
            )
            logger.info(
                "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    t + 1, mae, rmse, mape * 100
                )
            )
        mae, rmse, mape, _, _ = All_Metrics(
            y_pred, y_true, args.mae_thresh, args.mape_thresh
        )
        logger.info(
            "Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                mae, rmse, mape * 100
            )
        )
