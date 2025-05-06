import torch
import torch.nn as nn
import h5py
import random
from adabelief_pytorch import AdaBelief
from collections import deque
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, transformer: TransformerModel, isahp: ISAHpModel):
        super().__init__()
        self.transformer = transformer
        self.isahp = isahp
        transformer_out_dim = transformer.output_proj.out_features
        self.attention = nn.Linear(transformer.output_proj.out_features, 1)
        isahp_in_dim = isahp.input_dim + transformer_out_dim
        self.combined_isahp = ISAHpModel(input_dim=isahp_in_dim, hidden_dim=isahp.hidden_dim,
                                         output_dim=isahp.output_dim, num_layers=isahp.lstm.num_layers,
                                         dropout=isahp.dropout.p)

    def forward(self, x, mask):
        transformer_out = self.transformer(x, mask)
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
        # --- Állítható modellparaméterek ---
        t_depth = 4
        t_heads = 16
        isahp_depth = 8
        self.seq_len = 64
        self.max_events = 512
        self.feature_dim = 5
        self.embed_dim = 128
        self.sequence_window = deque(maxlen=self.seq_len + 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dropout_rate = 0.2
        self.max_grad_norm = 5.0
        self.patience = 3
        self.total_epochs = 15
        self.learn_rate = 1e-4
        self.factor = 0.1
        self.batch_size = 8

        self.multihead = CrossAttention(embed_dim=self.embed_dim, n_heads=t_depth * t_heads, dropout=self.dropout_rate).to(self.device)
        self.model_T = TransformerModel(num_layers=t_depth, max_events=self.max_events, num_heads=t_heads, dropout=self.dropout_rate).to(self.device)
        self.model_I = ISAHpModel(num_layers=isahp_depth, dropout=self.dropout_rate).to(self.device)
        self.model = CombinedModel(self.model_T, self.model_I).to(self.device)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.loss_fn = nn.MSELoss()

        self.optimizer = AdaBelief(self.model.parameters(), lr=self.learn_rate, weight_decay=1e-6, eps=1e-16, weight_decouple=True, rectify=True, print_change_log=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, factor=self.factor)
        self.initial_lr = self.optimizer.param_groups[0]['lr']

        self.train_data = []
        self.val_data = []
        self.token_map = {}

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.layer = 0
        self.corect_layer = 0
        self.last_processed_layer = 0
        self.global_epoch = 0
        self.global_step = 0

        self.model_save_path = f"multi_concat_model.pt"
        self.load_best_model()
        
    def load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {self.model_save_path}")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")

    def pad_batch(self, batch):
        """
        Paddolja az eseményláncokat, és maszkot készít az érvényes események jelölésére.
        """
        padded = torch.zeros(self.max_events, self.feature_dim, dtype=torch.float32)
        mask = torch.zeros(self.max_events, dtype=torch.bool)
        num_events = min(len(batch), self.max_events)
        if num_events > 0:
            stacked = torch.tensor(batch[:num_events], dtype=torch.float32)  # Egyszerűsítés: közvetlen tensor konverzió
            padded[:num_events] = stacked
            mask[:num_events] = 1
        return padded, mask

    def create_matrix(self, x, mask):
        """
        Átalakítja az eseményláncot egy mátrixszá, amely figyelembe veszi a delta értékeket és az események attribútumait.
        """
        x = x.squeeze(0)
        mask = mask.squeeze(0)
        T, E = x.shape[:2]
        t_idx = torch.arange(T).unsqueeze(1).expand(-1, E)
        e_idx = torch.arange(E).unsqueeze(0).expand(T, -1)
    
        f0, f1, f2, f3, f4, t_valid, e_valid = (x[..., 0][mask], x[..., 1][mask], x[..., 2][mask], x[..., 3][mask], x[..., 4][mask], t_idx[mask], e_idx[mask])
    
        df = pd.DataFrame({'row': list(zip(f1.cpu().numpy(), f2.cpu().numpy())), 'col': list(zip(t_valid.cpu().numpy(), f4.cpu().numpy())), 'value': f3.cpu().numpy()})
    
        unique_rows = {k: i for i, k in enumerate(df['row'].unique())}
        unique_cols = {k: i for i, k in enumerate(df['col'].unique())}
    
        row_map = torch.tensor(df['row'].map(unique_rows).to_numpy(), dtype=torch.long)
        col_map = torch.tensor(df['col'].map(unique_cols).to_numpy(), dtype=torch.long)
        values = torch.tensor(df['value'].to_numpy(), dtype=torch.float32)
    
        indices = torch.stack([row_map, col_map], dim=0)
        shape = (len(unique_rows), len(unique_cols))
        adj_mat = torch.sparse_coo_tensor(indices, values, size=shape).to_dense()
    
        return adj_mat, f0, f1, f2, f3, f4, t_valid, e_valid, row_map
    
    def rebuild_tensor_from_attention(self, attn_output, f0, f1, f3, f2, f4, t_valid, e_valid, row_map, target_shape=(1, 128, 512, 5)):
        """
        Az attention kimenetét visszaalakítja az eredeti tensor formátumába, figyelembe véve az attribútumokat.
        """
        device = attn_output.device
        B, T, E, F = target_shape
        output_tensor = torch.zeros((B, T, E, F), device=device)
    
        if not hasattr(self, "attention_projection"):
            self.attention_projection = nn.Linear(attn_output.shape[-1], F, bias=False).to(device)
    
        with torch.no_grad():
            attn_output = self.attention_projection(attn_output).squeeze(0)
            for i in range(attn_output.shape[0]):
                indices = (row_map == i).nonzero(as_tuple=True)[0]
                for j in indices:
                    t, e = t_valid[j].item(), e_valid[j].item()
                    output_tensor[0, t, e, 0] = attn_output[i, 3]  # Delta
                    output_tensor[0, t, e, 1] = f1[j]              # Price
                    output_tensor[0, t, e, 2] = f2[j]              # Entry ID
                    output_tensor[0, t, e, 3] = f3[j]              # Delta
                    output_tensor[0, t, e, 4] = f4[j]              # Side
        return output_tensor
    
    def process_layer(self):
        """
        Az eseményláncok feldolgozása, figyelembe véve a szabályok érvényesítését és az adat előkészítést.
        """
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
                        processed_rows = [[
                            event[0] / 100, event[1] / 100_000, event[2] / 1_000,
                            event[3] / 10_000_000, event[4]
                        ] for event in raw_rows]
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
                    input_tensor = torch.stack(input_batches).unsqueeze(0).to(self.device)
                    input_mask_tensor = torch.stack(input_masks).unsqueeze(0).to(self.device)
                    target_tensor = torch.stack(target_batches).unsqueeze(0).to(self.device)
                    adj_mat, f0, f1, f2, f3, f4, t_valid, e_valid, row_map = self.create_matrix(input_tensor, input_mask_tensor)
                    query = adj_mat.unsqueeze(0)
                    attn_output = self.multihead(query, query, query)
                    input_tensor = self.rebuild_tensor_from_attention(
                        attn_output, f0, f1, f2, f3, f4, t_valid, e_valid, row_map,
                        target_shape=input_tensor.shape
                    )
                    self.predict(input_tensor, input_mask_tensor)
                    for dataset, condition in [(self.train_data, random.random() < 0.8), (self.val_data, True)]:
                        if condition:
                            dataset.append((input_tensor, input_mask_tensor, target_tensor))
                    if len(self.train_data) > self.batch_size * 0.8 and len(self.val_data) > self.batch_size * 0.2:
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
        result_names = ["f0_time", "f1_price", "f2_entryid", "f3_delta", "f4_side"]
        thresholds = [0.01, 0.0001, 0.001, 0.00005, 0.0]
        results = {}
        acc_values = []
        for i, result_name in enumerate(result_names):
            abs_diff = torch.abs(pred[..., i] - targ[..., i])
            mae = abs_diff.mean().item()
            mse = (abs_diff ** 2).mean().item()
            acc = (abs_diff <= thresholds[i]).float().mean().item()
            results[f"{result_name}_mae"] = mae
            results[f"{result_name}_mse"] = mse
            results[f"{result_name}_acc"] = acc
            acc_values.append(acc)
        results["overall_mae"] = torch.abs(pred - targ).mean().item()
        results["overall_mse"] = (torch.abs(pred - targ) ** 2).mean().item()
        results["overall_acc"] = sum(acc_values) / len(acc_values)
        return results

    def build_token_map(self, batch_input):
        f0, f1, f2, f3, f4 = torch.chunk(batch_input, 5, dim=-1)
        f0_flat, f1_flat, f2_flat, f3_flat, f4_flat = (f0.view(-1), f1.view(-1), f2.view(-1), f3.view(-1), f4.view(-1))
        for idx in range(f2_flat.shape[0]):
            entry_id = int(round(f2_flat[idx].item() * 1000))
            curr_vals = {'f0': f0_flat[idx].item(), 'f1': f1_flat[idx].item(), 'f4': f4_flat[idx].item()}
            if entry_id not in self.token_map:
                self.token_map[entry_id] = {'f0': curr_vals['f0'], 'f1': curr_vals['f1'], 'f4': curr_vals['f4']}
            else:
                stored_vals = self.token_map[entry_id]
                curr_vals['f0'] = stored_vals['f0']
                curr_vals['f1'] = stored_vals['f1']
                curr_vals['f4'] = stored_vals['f4']
                self.token_map[entry_id] = {'f0': stored_vals['f0'], 'f1': stored_vals['f1'], 'f4': stored_vals['f4']}

    def select_predicted_f0(self, model_output):
        f0_pred = model_output[:, :, 0]
        f1_pred = model_output[:, :, 1]
        f2_pred = model_output[:, :, 2]
        f4_pred = model_output[:, :, 4]
        predict_list = {}
        loss = torch.tensor(0.0, device=f0_pred.device, requires_grad=True)
        for entry_id, values in self.token_map.items():
            indices = (f2_pred == entry_id).nonzero(as_tuple=True)
            if indices[0].numel() > 0:
                f0_values = f0_pred[indices]
                f1_values = f1_pred[indices]
                f4_values = f4_pred[indices]
                predict_list[entry_id] = {
                    'f0': f0_values.mean(),
                    'f1': f1_values.mean(),
                    'f4': f4_values.mean()
                }
            else:
                fake_pred = {
                    'f0': torch.tensor(0.0, device=f0_pred.device, requires_grad=True),
                    'f1': torch.tensor(0.0, device=f0_pred.device, requires_grad=True),
                    'f4': torch.tensor(0.0, device=f0_pred.device, requires_grad=True)
                }
                loss = loss + self.panalty(fake_pred, self.token_map[entry_id])
        return predict_list, loss

    def panalty(self, predict, true, loss=None):
        def apply_penalty(diff, thresholds, penalty_val):
            penalty = torch.tensor(0.0, device=diff.device)
            for scale in thresholds:
                if torch.any(diff > torch.tensor(scale, device=diff.device)):
                    penalty += penalty_val * (diff - scale).clamp(min=0)
                    break
            return penalty
        if loss is None:
            loss = torch.tensor(0.0, device=predict['f0'].device, requires_grad=True)
        penalty_f0 = 30.0
        penalty_group = 10.0
        thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        f0_diff = torch.abs(predict['f0'] - true['f0'])
        f1_diff = torch.abs(predict['f1'] - true['f1'])
        f4_diff = torch.abs(predict['f4'] - true['f4'])
        loss = loss + apply_penalty(f0_diff, thresholds, penalty_f0)
        loss = loss + apply_penalty(f1_diff, thresholds, penalty_group)
        loss = loss + apply_penalty(f4_diff, thresholds, penalty_group)
        return loss

    def custom_loss_fn(self, model_output, original_input):
        self.build_token_map(original_input)
        predict_list, loss = self.select_predicted_f0(model_output)
        known_ids = set(self.token_map.keys())
        predicted_ids = set(predict_list.keys())
        for entry_id, true_values in self.token_map.items():
            if entry_id in predict_list:
                pred_values = predict_list[entry_id]

                mismatched = any([
                    not torch.allclose(pred_values['f0'], true_values['f0'], atol=1e-3),
                    not torch.allclose(pred_values['f1'], true_values['f1'], atol=1e-3),
                    not torch.allclose(pred_values['f4'], true_values['f4'], atol=1e-3),
                ])
                if mismatched:
                    loss = self.panalty(pred_values, true_values, loss)
            else:
                if predicted_ids:
                    max_known = max(known_ids)
                    max_predicted = max(predicted_ids)
                    if max_predicted > max_known + len(predicted_ids - known_ids):
                        loss = loss + 0.1
        return loss

    def teach(self, input_tensor, input_mask_tensor, target_tensor):
        transformer_out = self.model_T(input_tensor, input_mask_tensor)
        target_lstm = target_tensor.mean(dim=2)
        input_lstm = target_lstm[:, :-1, :]
        target_lstm_out = target_lstm[:, 1:, :]
        mask_lstm = input_mask_tensor[:, :-1, :].any(dim=2)
        isahp_out = self.model_I(input_lstm, mask_lstm)
        loss_T = self.loss_fn(transformer_out, target_tensor)
        loss_I = self.loss_fn(isahp_out, target_lstm_out)
        rule_loss = self.custom_loss_fn(transformer_out, input_tensor)
        alpha, beta = 0.5, 0.5
        loss = alpha * (loss_I + loss_T) + beta * rule_loss
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
                if total_norm > self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                print(f"Gradient norm: {total_norm:.6f}, clipped to max: {self.max_grad_norm}")
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
                print(f"\033[93m--- Best_loss: {self.best_loss} SAVE MODEL ---\033[0m")
            elif epoch - self.best_epoch >= self.patience:
                print("\033[94m--- Early stopping triggered. ---\033[0m")
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
