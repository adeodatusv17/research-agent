import ast
import json
import logging
import re
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)

SUPPORTED_LOSS_ALIASES = {
    "cross_entropy": {"cross entropy", "cross-entropy", "ce", "nll"},
    "bce_with_logits": {"bce", "bcewithlogits", "binary cross entropy", "binary_cross_entropy"},
    "mse": {"mse", "mean squared error", "l2 loss"},
    "ctc": {"ctc", "ctc loss"},
}


def detect_model_family(model_name: str | None, architecture_text: str | None) -> str:
    haystack = f"{model_name or ''} {architecture_text or ''}".lower()
    if "conformer" in haystack:
        return "conformer"
    if "transformer" in haystack or "attention" in haystack:
        return "transformer"
    if any(token in haystack for token in ["lstm", "gru", "rnn"]):
        return "recurrent"
    if any(token in haystack for token in ["cnn", "convolution", "resnet"]):
        return "cnn"
    return "mlp"


def detect_task_type(dataset_name: str | None, training_objective: str | None, architecture_text: str | None) -> str:
    haystack = f"{dataset_name or ''} {training_objective or ''} {architecture_text or ''}".lower()
    if any(token in haystack for token in ["librispeech", "common voice", "speech", "asr", "transcript", "wer"]):
        return "speech_asr"
    if any(token in haystack for token in ["imagenet", "cifar", "mnist", "classification", "accuracy", "top-1"]):
        return "image_classification"
    if any(token in haystack for token in ["translation", "summarization", "seq2seq", "bleu"]):
        return "sequence_to_sequence"
    return "generic_supervised"


def normalize_loss_name(loss_name: str | None) -> str:
    lowered = (loss_name or "cross_entropy").strip().lower()
    for canonical, aliases in SUPPORTED_LOSS_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    if "ctc" in lowered:
        return "ctc"
    if "bce" in lowered:
        return "bce_with_logits"
    if "mse" in lowered:
        return "mse"
    return "cross_entropy"


def build_model_code(model_class: str, model_family: str, task_type: str) -> str:
    if model_family == "conformer":
        return f'''import torch
import torch.nn as nn


def _required(config: dict, path: str):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config field: {{path}}")
        current = current[key]
    if current is None:
        raise ValueError(f"Config field is unresolved: {{path}}")
    return current


class ConformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(hidden_dim)
        self.pointwise_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.pointwise_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.pointwise_in(conv_input)
        conv_output = self.glu(conv_output)
        conv_output = self.depthwise(conv_output)
        conv_output = self.batch_norm(conv_output)
        conv_output = torch.nn.functional.silu(conv_output)
        conv_output = self.pointwise_out(conv_output).transpose(1, 2)
        x = x + conv_output
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class {model_class}(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        dataset_cfg = config["dataset"]
        self.task_type = dataset_cfg.get("task_type", "{task_type}")
        input_dim = _required(config, "model.input_dim")
        hidden_dim = _required(config, "model.hidden_dim")
        output_dim = _required(config, "model.output_dim")
        num_layers = _required(config, "model.num_layers")
        num_heads = _required(config, "model.num_heads")
        dropout = model_cfg["dropout"]

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ConformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        logits = self.output_head(x)
        if self.task_type == "speech_asr":
            return logits
        return logits.mean(dim=1)
'''

    if model_family == "transformer":
        return f'''import torch
import torch.nn as nn


def _required(config: dict, path: str):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config field: {{path}}")
        current = current[key]
    if current is None:
        raise ValueError(f"Config field is unresolved: {{path}}")
    return current


class {model_class}(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        dataset_cfg = config["dataset"]
        self.task_type = dataset_cfg.get("task_type", "{task_type}")
        input_dim = _required(config, "model.input_dim")
        hidden_dim = _required(config, "model.hidden_dim")
        output_dim = _required(config, "model.output_dim")
        num_layers = _required(config, "model.num_layers")
        num_heads = _required(config, "model.num_heads")
        dropout = model_cfg["dropout"]
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.encoder(x)
        logits = self.output_head(x)
        if self.task_type == "speech_asr":
            return logits
        return logits.mean(dim=1)
'''

    if model_family == "recurrent":
        return f'''import torch
import torch.nn as nn


def _required(config: dict, path: str):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config field: {{path}}")
        current = current[key]
    if current is None:
        raise ValueError(f"Config field is unresolved: {{path}}")
    return current


class {model_class}(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        dataset_cfg = config["dataset"]
        self.task_type = dataset_cfg.get("task_type", "{task_type}")
        input_dim = _required(config, "model.input_dim")
        hidden_dim = _required(config, "model.hidden_dim")
        output_dim = _required(config, "model.output_dim")
        num_layers = _required(config, "model.num_layers")
        dropout = model_cfg["dropout"]
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        logits = self.output_head(encoded)
        if self.task_type == "speech_asr":
            return logits
        return logits[:, -1, :]
'''

    if model_family == "cnn":
        return f'''import torch
import torch.nn as nn


def _required(config: dict, path: str):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config field: {{path}}")
        current = current[key]
    if current is None:
        raise ValueError(f"Config field is unresolved: {{path}}")
    return current


class {model_class}(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        input_dim = _required(config, "model.input_dim")
        hidden_dim = _required(config, "model.hidden_dim")
        output_dim = _required(config, "model.output_dim")
        num_layers = _required(config, "model.num_layers")
        dropout = model_cfg["dropout"]

        layers = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.encoder(x).mean(dim=-1)
        return self.output_head(encoded)
'''

    return f'''import torch
import torch.nn as nn


def _required(config: dict, path: str):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required config field: {{path}}")
        current = current[key]
    if current is None:
        raise ValueError(f"Config field is unresolved: {{path}}")
    return current


class {model_class}(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        input_dim = _required(config, "model.input_dim")
        hidden_dim = _required(config, "model.hidden_dim")
        output_dim = _required(config, "model.output_dim")
        num_layers = _required(config, "model.num_layers")
        dropout = model_cfg["dropout"]
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        return self.network(x)
'''


def build_dataset_code(task_type: str) -> str:
    if task_type == "speech_asr":
        return '''import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    def __init__(self, config: dict, split: str = "train"):
        self.config = config
        dataset_cfg = config["dataset"]
        model_cfg = config["model"]
        data_path = Path(dataset_cfg["path"])
        manifest_path = data_path / f"{split}_manifest.jsonl"
        self.records = []
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self.records.append(json.loads(line))
        else:
            sequence_length = dataset_cfg["sequence_length"]
            input_dim = model_cfg["input_dim"]
            target_length = min(dataset_cfg["target_length"], sequence_length)
            self.records.append(
                {
                    "features": np.zeros((sequence_length, input_dim), dtype=np.float32).tolist(),
                    "targets": [0 for _ in range(target_length)],
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if isinstance(record["features"], str):
            inputs = np.load(record["features"])
        else:
            inputs = np.asarray(record["features"], dtype=np.float32)
        targets = np.asarray(record["targets"], dtype=np.int64)
        input_lengths = np.asarray(inputs.shape[0], dtype=np.int64)
        target_lengths = np.asarray(len(targets), dtype=np.int64)
        return (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
        )
'''

    if task_type == "image_classification":
        return '''from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    def __init__(self, config: dict, split: str = "train"):
        self.config = config
        dataset_cfg = config["dataset"]
        model_cfg = config["model"]
        data_path = Path(dataset_cfg["path"])
        self.data_file = data_path / f"{split}_data.npy"
        self.label_file = data_path / f"{split}_labels.npy"
        if self.data_file.exists() and self.label_file.exists():
            self.inputs = np.load(self.data_file)
            self.labels = np.load(self.label_file)
        else:
            sequence_length = dataset_cfg["sequence_length"]
            input_dim = model_cfg["input_dim"]
            output_dim = model_cfg["output_dim"]
            self.inputs = np.zeros((8, sequence_length, input_dim), dtype=np.float32)
            self.labels = np.zeros((8,), dtype=np.int64)
            if output_dim == 1:
                self.labels = np.zeros((8,), dtype=np.int64)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        x = torch.tensor(self.inputs[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y
'''

    return '''from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    def __init__(self, config: dict, split: str = "train"):
        self.config = config
        dataset_cfg = config["dataset"]
        model_cfg = config["model"]
        data_path = Path(dataset_cfg["path"])
        self.data_file = data_path / f"{split}_data.npy"
        self.label_file = data_path / f"{split}_labels.npy"
        if self.data_file.exists() and self.label_file.exists():
            self.inputs = np.load(self.data_file)
            self.labels = np.load(self.label_file)
        else:
            self.inputs = np.zeros((8, dataset_cfg["sequence_length"], model_cfg["input_dim"]), dtype=np.float32)
            self.labels = np.zeros((8,), dtype=np.int64)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        x = torch.tensor(self.inputs[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y
'''


def build_utils_code() -> str:
    return '''from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
'''


def build_train_code(model_class: str, task_type: str) -> str:
    return f'''import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import GeneratedDataset
from model import {model_class}
from utils.config_loader import load_config


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(config: dict, model: nn.Module):
    training_cfg = config["training"]
    name = training_cfg["optimizer"].lower()
    params = training_cfg.get("optimizer_params", {{}})
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **params)
    return torch.optim.Adam(model.parameters(), **params)


def build_scheduler(config: dict, optimizer):
    scheduler_cfg = config["scheduler"]
    scheduler_type = scheduler_cfg["type"].lower()
    if scheduler_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )
    return None


def build_loss(config: dict):
    loss_name = (config["training"].get("loss") or "cross_entropy").lower()
    if loss_name == "ctc":
        return "ctc", nn.CTCLoss(blank=0, zero_infinity=True)
    if loss_name == "mse":
        return "mse", nn.MSELoss()
    if loss_name == "bce_with_logits":
        return "bce_with_logits", nn.BCEWithLogitsLoss()
    return "cross_entropy", nn.CrossEntropyLoss()


def validate(model: nn.Module, val_loader, criterion, loss_kind: str, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if loss_kind == "ctc":
                inputs, targets, input_lengths, target_lengths = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                total_loss += criterion(
                    log_probs,
                    targets,
                    input_lengths.to(device),
                    target_lengths.to(device),
                ).item()
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, targets).item()
    return total_loss / max(1, len(val_loader))


def main():
    logging.basicConfig(level=logging.INFO)
    config = load_config("config.yaml")
    set_seed(config["training"]["seed"])

    dataset = GeneratedDataset(config, split="train")
    val_size = max(1, int(len(dataset) * config["dataset"]["val_split"]))
    train_size = max(1, len(dataset) - val_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    device_name = "cuda" if config["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=device_name == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=device_name == "cuda",
    )

    model = {model_class}(config).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    loss_kind, criterion = build_loss(config)

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch={{epoch + 1}}"):
            optimizer.zero_grad()
            if loss_kind == "ctc":
                inputs, targets, input_lengths, target_lengths = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(
                    log_probs,
                    targets,
                    input_lengths.to(device),
                    target_lengths.to(device),
                )
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)

            loss.backward()
            gradient_clip = config["training"]["gradient_clip"]
            if gradient_clip > 0:
                clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            global_step += 1

            if global_step % config["training"]["log_every"] == 0:
                logger.info("train_step step=%s loss=%s", global_step, float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % config["training"]["eval_every"] == 0:
            val_loss = validate(model, val_loader, criterion, loss_kind, device)
            logger.info("validation epoch=%s val_loss=%s", epoch + 1, val_loss)

        if config["training"]["save_checkpoint"] and (epoch + 1) % config["training"]["checkpoint_every"] == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{{epoch + 1}}.pt")


if __name__ == "__main__":
    main()
'''


def build_requirements_txt() -> str:
    return "\n".join(
        [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",
        ]
    )


def validate_generated_artifact(
    config_yaml: str,
    model_code: str,
    dataset_code: str,
    train_code: str,
    utils_code: str,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        config = yaml.safe_load(config_yaml) or {}
    except Exception as exc:
        raise ValueError(f"config_yaml_invalid: {exc}") from exc

    for filename, source in [
        ("model.py", model_code),
        ("dataset.py", dataset_code),
        ("train.py", train_code),
        ("utils/config_loader.py", utils_code),
    ]:
        try:
            compile(source, filename, "exec")
        except Exception as exc:
            raise ValueError(f"{filename}_compile_failed: {exc}") from exc

    loss_name = str((((config.get("training") or {}).get("loss")) or "cross_entropy")).lower()
    normalized_loss = normalize_loss_name(loss_name)
    if normalized_loss == "ctc" and "nn.CTCLoss" not in train_code:
        errors.append("ctc_loss_requested_but_train_code_has_no_ctc_support")

    if "print(" in train_code:
        warnings.append("train_code_uses_print_logging")

    required_fields = [
        "model.name",
        "dataset.name",
        "training.optimizer",
        "training.loss",
    ]
    for path in required_fields:
        current = config
        for key in path.split("."):
            current = current.get(key) if isinstance(current, dict) else None
        if current in (None, ""):
            errors.append(f"missing_required_config_field:{path}")

    unresolved_fields = []
    for path in [
        "model.input_dim",
        "model.output_dim",
        "model.hidden_dim",
        "model.num_layers",
    ]:
        current = config
        for key in path.split("."):
            current = current.get(key) if isinstance(current, dict) else None
        if current is None:
            unresolved_fields.append(path)
    if unresolved_fields:
        warnings.append(f"unresolved_model_fields:{','.join(unresolved_fields)}")

    return {
        "errors": errors,
        "warnings": warnings,
        "normalized_loss": normalized_loss,
    }
