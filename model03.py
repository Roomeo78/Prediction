import torch
import torch.nn as nn
import h5py
import random
import torch.optim as optim
from adabelief_pytorch import AdaBelief
from collections import deque
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import math
from torch.optim.lr_scheduler import LambdaLR

class ISAHpModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, output_dim=5, num_layers=2, dropout=0.2):
        super(ISAHpModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).long()
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        fc1_out = self.relu(self.fc1(lstm_out))
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        return fc2_out

class TransformerModel(nn.Module):
    def __init__(self, feature_dim=5, model_dim=256, num_heads=4, ff_dim=512, num_layers=2, max_events=512, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, feature_dim)
        self.max_events = max_events
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, E, F = x.shape
        x = x.view(B * T, E, F)
        mask = ~mask.view(B * T, E)
        x = self.input_proj(x)
        encoded = self.encoder(self.dropout(x), src_key_padding_mask=mask)
        out = self.output_proj(encoded)
        return out.view(B, T, E, F)

class CombinedModel(nn.Module):
    def __init__(self, transformer: TransformerModel, isahp: ISAHpModel, method='concat'):
        super().__init__()
        self.transformer = transformer
        self.isahp = isahp
        transformer_out_dim = transformer.output_proj.out_features
        self.method = method
        if method == 'concat+linear_attn':
            self.attention = nn.Linear(transformer.output_proj.out_features, 1)
            isahp_in_dim = isahp.input_dim + transformer_out_dim
            self.combined_isahp = ISAHpModel(input_dim=isahp_in_dim, hidden_dim=isahp.hidden_dim,
                                             output_dim=isahp.output_dim, num_layers=isahp.lstm.num_layers,
                                             dropout=isahp.dropout.p)
        elif method == 'concat+multihead':
            self.attention = nn.MultiheadAttention(embed_dim=transformer_out_dim, num_heads=1, batch_first=True)
            isahp_in_dim = isahp.input_dim + transformer_out_dim
            self.combined_isahp = ISAHpModel(input_dim=isahp_in_dim, hidden_dim=isahp.hidden_dim,
                                             output_dim=isahp.output_dim, num_layers=isahp.lstm.num_layers,
                                             dropout=isahp.dropout.p)

        elif method == 'multihead+concat':
            self.attention = nn.Linear(transformer.output_proj.out_features, 1)
            isahp_in_dim = isahp.input_dim + transformer_out_dim
            self.combined_isahp = ISAHpModel(input_dim=isahp_in_dim, hidden_dim=isahp.hidden_dim,
                                             output_dim=isahp.output_dim, num_layers=isahp.lstm.num_layers,
                                             dropout=isahp.dropout.p)
    def forward(self, x, mask):
        transformer_out = self.transformer(x, mask)
        if self.method == 'concat+linear_attn':
            attention_weights = torch.softmax(self.attention(transformer_out), dim=2)
            attended_output = torch.sum(attention_weights * transformer_out, dim=2)
            x_lstm = x.mean(dim=2)
            combined_input = torch.cat((x_lstm, attended_output), dim=-1)
            mask_lstm = mask.any(dim=2)
            return self.combined_isahp(combined_input, mask_lstm)
        elif self.method == 'concat+multihead':
            transformer_out = transformer_out.squeeze(0)
            attn_output, _ = self.attention(transformer_out, transformer_out, transformer_out)
            x_lstm = x.mean(dim=0)
            combined_input = torch.cat((x_lstm, attn_output), dim=-1)
            mask_lstm = mask.any(dim=2)
            return self.combined_isahp(combined_input, mask_lstm)
        elif self.method == 'multihead+concat':
            attention_weights = torch.softmax(self.attention(transformer_out), dim=2)
            attended_output = torch.sum(attention_weights * transformer_out, dim=2)
            x_lstm = x.mean(dim=2)
            combined_input = torch.cat((x_lstm, attended_output), dim=-1)
            mask_lstm = mask.any(dim=2)
            return self.combined_isahp(combined_input, mask_lstm)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        attn_output, attn_weights = self.attn(queries, keys, values, attn_mask=None)
        attn_output = self.dropout(attn_output)
        return attn_output


class Predict:
    def __init__(self, log_dir=None):

        self.seq_len = 64
        self.max_events = 512
        self.feature_dim = 5
        self.embed_dim = 128
        self.sequence_window = deque(maxlen=self.seq_len + 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Állítható modellparaméterek ---
        t_depth = 4
        t_heads = 16
        isahp_depth = 8

        self.dropout_rate = 0.2
        self.combination = 'multihead+concat'
        self.multihead = CrossAttention(embed_dim=self.embed_dim, n_heads=t_depth*t_heads, dropout=self.dropout_rate).to(self.device)
        self.model_T = TransformerModel(num_layers=t_depth, max_events=self.max_events, num_heads=t_heads, dropout=self.dropout_rate).to(self.device)
        self.model_I = ISAHpModel(num_layers=isahp_depth, dropout=self.dropout_rate).to(self.device)
        self.model = CombinedModel(self.model_T, self.model_I, method=self.combination).to(self.device)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.loss_fn = nn.MSELoss()
        self.patience = 3
        self.total_epochs = 15
        self.warmup_epochs = 5
        #self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        self.optimizer = AdaBelief(self.model.parameters(), lr=1e-4, weight_decay=1e-6, eps=1e-16, weight_decouple=True, rectify=True, print_change_log=False)

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            return 0.5 * (1 + math.cos((epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs) * math.pi))

        #self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=0.1)
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.train_data = []
        self.val_data = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.model_save_path = f"{self.combination}best_model.pt"
        self.load_best_model()
        self.layer = 0
        self.corect_layer = 0
        self.last_processed_layer = 0
        self.global_epoch = 0
        self.global_step = 0
    def load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {self.model_save_path}")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")

    def pad_batch(self, batch):
        padded = torch.zeros(self.max_events, self.feature_dim, dtype=torch.float32)
        mask = torch.zeros(self.max_events, dtype=torch.bool)
        num_events = min(len(batch), self.max_events)
        if num_events > 0:
            stacked = torch.stack([torch.tensor(event, dtype=torch.float32) for event in batch[:num_events]])
            padded[:num_events] = stacked
            mask[:num_events] = 1
        return padded, mask

    def create_matrix(self, x, mask):
        x = x.squeeze(0)
        mask = mask.squeeze(0)
        T, E = x.shape[:2]
        t_idx = torch.arange(T).unsqueeze(1).expand(-1, E)
        e_idx = torch.arange(E).unsqueeze(0).expand(T, -1)
        f0, f1, f2, f3, f4, t_valid, e_valid = x[..., 0][mask], x[..., 1][mask], x[..., 2][mask], x[..., 3][mask], x[..., 4][mask], t_idx[mask], e_idx[mask]
        df = pd.DataFrame({'row': list(zip(f1.cpu().numpy(), f2.cpu().numpy())), 'col': list(zip(t_valid.cpu().numpy(), f4.cpu().numpy())), 'value': f3.cpu().numpy()})
        unique_rows, unique_cols = {k: i for i, k in enumerate(df['row'].unique())}, {k: i for i, k in enumerate(df['col'].unique())}
        row_map, col_map, values = torch.tensor(df['row'].map(unique_rows).to_numpy(), dtype=torch.long), torch.tensor(df['col'].map(unique_cols).to_numpy(), dtype=torch.long), torch.tensor(df['value'].to_numpy(), dtype=torch.float32)
        indices = torch.stack([row_map, col_map], dim=0)
        shape = (len(unique_rows), len(unique_cols))
        adj_mat = torch.sparse_coo_tensor(indices, values, size=shape)
        adj_mat = adj_mat.to_dense()
        return adj_mat, f0, f1, f2, f3, f4, t_valid, e_valid, row_map

    def rebuild_tensor_from_attention(self, attn_output, f0, f1, f3, f2, f4, t_valid, e_valid, row_map, target_shape=(1, 128, 512, 5)):
        device = attn_output.device
        B, T, E, F = target_shape
        output_tensor = torch.zeros((B, T, E, F), device=device)
        if not hasattr(self, "attention_projection"):
            self.attention_projection = nn.Linear(attn_output.shape[-1], F, bias=False).to(device)
        with torch.no_grad():
            attn_output = self.attention_projection(attn_output)
            attn_output = attn_output.squeeze(0)
            for i in range(attn_output.shape[0]):
                indices = (row_map == i).nonzero(as_tuple=True)[0]
                for j in indices:
                    t, e = t_valid[j].item(), e_valid[j].item()
                    output_tensor[0, t, e, 0] = attn_output[i, 3]
                    output_tensor[0, t, e, 1] = f1[j]
                    output_tensor[0, t, e, 2] = f2[j]
                    output_tensor[0, t, e, 3] = f3[j]
                    output_tensor[0, t, e, 4] = f4[j]
        return output_tensor

    def process_layer(self):
        while True:
            if self.layer == self.last_processed_layer:
                self.layer += 1
                print(f"\033[91m!!! {self.layer}. iteráció !!!\033[0m")
                with h5py.File("orderbook_05011132.h5", "r") as orderbook:
                    i = self.layer
                    collected = 0
                    while collected < self.seq_len + 1:
                        raw_rows = orderbook[str(i)][:]
                        if len(raw_rows) == 0:
                            i += 1
                            continue
                        processed_rows = [[event[0] / 100, event[1] / 100_000, event[2] / 1_000, event[3] / 10_000_000, event[4]] for event in raw_rows]
                        self.sequence_window.append(processed_rows)
                        collected += 1
                        i += 1
                    input_batches, input_masks, target_batches = [], [], []
                    for t in range(self.seq_len):
                        input_padded, input_mask = self.pad_batch(self.sequence_window[t])
                        target_padded, _ = self.pad_batch(self.sequence_window[t + 1])
                        input_batches.append(input_padded)
                        input_masks.append(input_mask)
                        target_batches.append(target_padded)
                    #next_batch = [[token[0], token[1], token[2], token[3], token[4]] for token in self.sequence_window[-1]]
                    #print(f"Cél érték next_batch: {next_batch}")
                    input_tensor = torch.stack(input_batches).unsqueeze(0).to(self.device)
                    input_mask_tensor = torch.stack(input_masks).unsqueeze(0).to(self.device)
                    target_tensor = torch.stack(target_batches).unsqueeze(0).to(self.device)
                    if self.combination == "multihead+concat":
                        adj_mat, f0, f1, f2, f3, f4, t_valid, e_valid, row_map = self.create_matrix(input_tensor, input_mask_tensor)
                        query = adj_mat.unsqueeze(0)
                        attn_output = self.multihead(query, query, query)
                        input_tensor = self.rebuild_tensor_from_attention(
                            attn_output,
                            f0, f1, f2, f3, f4, t_valid, e_valid, row_map,
                            target_shape=input_tensor.shape
                        )
                    self.predict(input_tensor, input_mask_tensor)
                    if len(self.val_data) == 0:
                        self.val_data.append((input_tensor, input_mask_tensor, target_tensor))
                    elif len(self.train_data) < 10 or random.random() < 0.8:
                        if not any(torch.equal(input_tensor, item[0]) and torch.equal(input_mask_tensor, item[1]) and torch.equal(target_tensor, item[2]) for item in self.train_data):
                            self.train_data.append((input_tensor, input_mask_tensor, target_tensor))
                        else:
                            if not any(torch.equal(input_tensor, item[0]) and torch.equal(input_mask_tensor, item[1]) and torch.equal(target_tensor, item[2]) for item in self.val_data):
                                self.val_data.append((input_tensor, input_mask_tensor, target_tensor))
                    else:
                        if not any(torch.equal(input_tensor, item[0]) and torch.equal(input_mask_tensor, item[1]) and torch.equal(target_tensor, item[2]) for item in self.val_data):
                            self.val_data.append((input_tensor, input_mask_tensor, target_tensor))
                        else:
                            pass
                    if len(self.train_data) > 5 and len(self.val_data) > 1:
                        self.train()
                        self.train_data = []
                        self.val_data = []
                        self.corect_layer += 1
                        self.last_processed_layer = self.corect_layer
            self.layer = self.last_processed_layer

    def predict(self, input_tensor, input_mask_tensor):
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(input_tensor, input_mask_tensor)
            #next_batch = [[token[0], token[1], token[2], token[3], token[4]] for token in predicted if not torch.all(token == 0)]
        #print(f"Predikciós érték: {next_batch}")

    def evaluate_model(self, predictions, targets):
        pred = predictions.detach().cpu()
        targ = targets.detach().cpu()

        results = {}

        # f0: time of first appearance (float), normalizált /100
        f0_tol = 0.01  # 1 időegység
        f0_diff = torch.abs(pred[..., 0] - targ[..., 0])
        f0_acc = (f0_diff <= f0_tol).float().mean().item()
        results["f0_time_mae"] = f0_diff.mean().item()
        results["f0_time_mse"] = (f0_diff ** 2).mean().item()  # MSE hozzáadása
        results["f0_time_acc"] = f0_acc

        # f1: price, normalizált /100_000
        f1_tol = 0.0001  # ~10 egységnyi tolerancia eredetiben
        f1_diff = torch.abs(pred[..., 1] - targ[..., 1])
        f1_acc = (f1_diff <= f1_tol).float().mean().item()
        results["f1_price_mae"] = f1_diff.mean().item()
        results["f1_price_mse"] = (f1_diff ** 2).mean().item()  # MSE hozzáadása
        results["f1_price_acc"] = f1_acc

        # f2: entry_id, diszkrét, normalizált /1000
        f2_tol = 0.001  # pontos egyezést várunk
        f2_diff = torch.abs(pred[..., 2] - targ[..., 2])
        f2_acc = (f2_diff <= f2_tol).float().mean().item()
        results["f2_entryid_mae"] = f2_diff.mean().item()
        results["f2_entryid_mse"] = (f2_diff ** 2).mean().item()  # MSE hozzáadása
        results["f2_entryid_acc"] = f2_acc

        # f3: delta, normalizált /10_000_000, min 8 számjegy → változások elég nagyok
        f3_tol = 0.00005  # ~500 egység delta eredetiben
        f3_diff = torch.abs(pred[..., 3] - targ[..., 3])
        f3_acc = (f3_diff <= f3_tol).float().mean().item()
        results["f3_delta_mae"] = f3_diff.mean().item()
        results["f3_delta_mse"] = (f3_diff ** 2).mean().item()
        results["f3_delta_acc"] = f3_acc

        # f4: side, 1 vagy 2 → pontos egyezés kell
        f4_pred = pred[..., 4].round()
        f4_true = targ[..., 4]
        f4_acc = (f4_pred == f4_true).to(torch.float32).mean().item()

        results["f4_side_acc"] = f4_acc
        results["f4_side_mae"] = torch.abs(f4_pred - f4_true).float().mean().item()
        results["f4_side_mse"] = ((f4_pred - f4_true) ** 2).mean().item()
        results["overall_acc"] = (f0_acc + f1_acc + f2_acc + f3_acc + f4_acc) / 5

        # Általános MSE (összes jellemzőre)
        all_diff = torch.abs(pred - targ)
        results["overall_mse"] = (all_diff ** 2).mean().item()

        # Általános MAE (összes jellemzőre)
        all_mae = torch.abs(pred - targ)
        results["overall_mae"] = all_mae.mean().item()

        return results

    def build_token_map(self, batch_input):
        f0, f1, f2, f3, f4 = torch.chunk(batch_input, 5, dim=-1)
        token_map = {}  # f2 → {fix f2->f0, fix f2->f1, fix f2->f4}
        f0_flat, f1_flat, f2_flat, f3_flat, f4_flat = f0.view(-1), f1.view(-1), f2.view(-1), f3.view(-1), f4.view(-1)
        for idx in range(f2_flat.shape[0]):
            entry_id = f2_flat[idx].item()
            if entry_id not in token_map:
                token_map[entry_id] = {'f0': f0_flat[idx].item(), 'f1': f1_flat[idx].item(), 'f4': f4_flat[idx].item()}
        print(f"token_map: {token_map}")
        return token_map

    def select_predicted_f0(self, model_output, token_map):
        f0_pred = model_output[:, :, 0]
        f2_list = {}
        for entry_id, values in token_map.items():
            f0_value = values['f0']
            indices = [i for i, entry in enumerate(f0_value) if entry == entry_id]
            f2_list[entry_id] = f0_pred[indices]
        return f2_list

    def custom_loss_fn(self, model_output, original_input):
        token_map = self.build_token_map(original_input)
        f2_list = self.select_predicted_f0(model_output, token_map)
        f3_pred = model_output[:, :, 3]
        loss = 0.0
        for entry_id, values in f2_list.items():
            f0_list = values['f0']
            f1_list = values['f1']
            f4_list = values['f4']
            f2_tensor = torch.tensor(f0_list, f1_list, f4_list, f3_pred)
            loss = self.loss_fn(model_output, f2_tensor)
        return loss

    def teach(self, input_tensor, input_mask_tensor, target_tensor):
        transformer_out = self.model_T(input_tensor, input_mask_tensor)
        #f0, f1, f2, f3, f4 = [input_tensor[..., i] for i in range(5)]
        #new_f0 = transformer_out[..., 0]
        #combined = torch.stack([new_f0, f1, f2, f3, f4], dim=-1)
        target_lstm = target_tensor.mean(dim=2)
        input_lstm = target_lstm[:, :-1, :]
        target_lstm_out = target_lstm[:, 1:, :]
        mask_lstm = input_mask_tensor[:, :-1, :].any(dim=2)
        isahp_out = self.model_I(input_lstm, mask_lstm)
        #loss = self.custom_loss_fn(combined, input_tensor)
        loss_T = self.loss_fn(transformer_out, target_tensor)
        loss_I = self.loss_fn(isahp_out, target_lstm_out)
        loss = loss_I + loss_T
        return loss, isahp_out, target_lstm_out

    def train(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        for epoch in range(self.total_epochs):
            epoch_loss = 0
            self.model.train()
            for input_tensor, input_mask_tensor, target_tensor in self.train_data:
                loss, _, _ = self.teach(input_tensor, input_mask_tensor, target_tensor)
                loss.backward()
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient norm: {total_norm:.6f}")
                max_grad_norm = 5.0
                if total_norm > max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                    print(f"Gradient norm exceeded {max_grad_norm}, clipped to max.")
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()

            self.writer.add_scalar("epoch_loss", epoch_loss, self.global_epoch)
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], self.global_epoch)
            val_loss, metrics = self.validate()
            self.writer.add_scalar("val_loss", val_loss, self.global_epoch)
            for key, value in metrics.items():
                if "_" in key:
                    field, metric = key.split("_", 1)
                    self.writer.add_scalar(f"val/{field}/{metric}", value, self.global_epoch)
                else:
                    self.writer.add_scalar(f"val/{key}", value, self.global_epoch)

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Epoch {epoch + 1} - Training Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.6e}")
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"Best_loss: {self.best_loss} SAVE MODEL")
            elif epoch - self.best_epoch >= self.patience:
                print("Early stopping triggered.")
                break
            self.scheduler.step(val_loss)
            new_dropout = max(0.0, self.dropout_rate * (0.9 ** epoch))
            for mod in self.model.modules():
                if isinstance(mod, nn.Dropout):
                    mod.p = new_dropout
            self.global_epoch += 1
            print(f"\033[92m--- Global Epoch {self.global_epoch} ---\033[0m")

    def validate(self):
        self.model.eval()
        metrics = {}
        val_loss = 0
        with torch.no_grad():
            for input_tensor, input_mask_tensor, target_tensor in self.val_data:
                loss, output_I, target_lstm_out = self.teach(input_tensor, input_mask_tensor, target_tensor)
                val_loss += loss.item()
                results = self.evaluate_model(output_I, target_lstm_out)
                for key, value in results.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        avg_val_loss = val_loss / len(self.val_data)
        print(f"Loss: {avg_val_loss:.6f}")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.6f}")
        return avg_val_loss, avg_metrics

if __name__ == "__main__":
    print("Loading modell...")
    Predict().process_layer()
