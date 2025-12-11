import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    Blocco base della TCN.
    Contiene due layer convoluzionali causali dilatati con connessione residua.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()

        # Padding causale: guarda solo nel passato
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Connessione residua (1x1 conv se cambiano i canali)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # Ramo principale
        out = self.conv1(x)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Connessione residua
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_out(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network per PV Power Forecasting.

    Usa convoluzioni causali dilated per catturare dipendenze temporali
    a diverse scale, senza data leakage.
    """

    def __init__(self, model_config: dict):
        """
        Args:
            model_config: Dizionario con i parametri del modello (da config.py).
        """
        super(TCN, self).__init__()

        # --- Lettura parametri da config (non più da train_loader) ---
        self.input_size = model_config.get("input_size", 25)

        # Parametri da configurazione (come gli altri modelli)
        self.output_size = model_config.get("output_size", 24)
        self.hidden_size = model_config.get("hidden_size", 64)
        self.num_layers = model_config.get("num_layers", 4)
        self.kernel_size = model_config.get("kernel_size", 3)
        self.dropout = model_config.get("dropout", 0.2)

        print(f"TCN - Features: {self.input_size}")

        # --- COSTRUZIONE DELLA RETE TCN ---
        layers = []
        for i in range(self.num_layers):
            # La dilatazione cresce esponenzialmente: 1, 2, 4, 8...
            dilation = 2**i

            # Primo layer: input_size -> hidden_size
            # Layer successivi: hidden_size -> hidden_size
            in_ch = self.input_size if i == 0 else self.hidden_size
            out_ch = self.hidden_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # Layer di output: hidden_size -> horizon (24)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass della TCN.

        Input x:  (Batch, Lookback, Input_Size) -> es. (64, 48, 10)
        Output:   (Batch, Horizon, 1) -> es. (64, 24, 1)
        """
        # --- PASSO 1: RESHAPE PER CONV1D ---
        # Conv1d si aspetta (Batch, Channels, Length)
        x = x.permute(0, 2, 1)  # (Batch, Input_Size, Lookback)

        # --- PASSO 2: ELABORAZIONE TCN ---
        out = self.network(x)  # (Batch, hidden_size, Lookback)

        # --- PASSO 3: SELEZIONE ULTIMO TIMESTEP ---
        # Prendiamo l'ultimo timestep (causale, contiene info su tutta la storia)
        out = out[:, :, -1]  # (Batch, hidden_size)

        # --- PASSO 4: PREDIZIONE ---
        prediction = self.fc(out)  # (Batch, Horizon)

        # --- PASSO 5: RESHAPE PER IL TRAINING LOOP ---
        return prediction.unsqueeze(-1)  # (Batch, Horizon, 1)
