import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM per PV Power Forecasting.

    """

    def __init__(self, model_config: dict):
        super(LSTM, self).__init__()

        # --- Lettura parametri da config (non più da train_loader) ---
        self.input_size = model_config.get("input_size", 24)

        # Il resto delle configurazioni rimane uguale
        self.hidden_size = model_config.get("hidden_size", 64)
        self.output_size = model_config.get("output_size", 24)
        self.num_layers = model_config.get("num_layers", 1)
        self.dropout = model_config.get("dropout", 0.0)
        self.bidirectional = model_config.get("bidirectional", False)

        print(
            f"LSTM - Features: {self.input_size}, Hidden: {self.hidden_size}, Layers: {self.num_layers}, Bidir: {self.bidirectional}"
        )

        # bidirectional: processa il lookback in entrambe le direzioni.
        # È valido per forecasting perché entrambe le direzioni vedono solo
        # dati storici (lookback), non il target futuro. Empiricamente spesso
        # non migliora molto, ma può essere testato.
        #
        # batch_first=True è richiesto per consistenza con il DataLoader che
        # restituisce tensori (Batch, Lookback, Features) e con gli altri modelli.

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True,  # Richiesto per shape (Batch, Lookback, Features)
        )

        # Con bidirectional=True, l'output ha dimensione hidden_size * 2
        fc_input_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        # LayerNorm per stabilizzare il training
        # Normalizza le attivazioni, permette learning rate più alti
        self.layer_norm = nn.LayerNorm(fc_input_size)

        self.fc = nn.Linear(fc_input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: è il batch di input.
           Shape attesa: (Batch_Size, Lookback, Input_Size) -> es. (64, 48, 25)
        """

        # --- PASSO A: ELABORAZIONE TEMPORALE ---
        # Passiamo tutto il "filmato" (x) dentro la LSTM.
        # lstm_out: Contiene lo stato della memoria per OGNI istante temporale (t-48, t-47... t-1).
        #           Shape: (Batch_Size, Lookback, Hidden_Size) -> es. (64, 48, 64)
        #           Con bidirectional: (Batch_Size, Lookback, Hidden_Size * 2)
        # _ : Sono gli stati nascosti finali (h_n, c_n). Non ci servono direttamente perché sono già contenuti in lstm_out.
        lstm_out, _ = self.lstm(x)

        # --- PASSO B: SELEZIONE DEGLI STATI FINALI ---
        if self.bidirectional:
            # Con BiLSTM, dobbiamo prendere gli stati finali corretti:
            # - Forward: ultimo timestep (indice -1) contiene info da t=0 a T
            # - Backward: primo timestep (indice 0) contiene info da T a 0
            # PyTorch concatena [forward, backward] sull'ultima dimensione
            out_forward = lstm_out[:, -1, : self.hidden_size]  # (Batch, Hidden)
            out_backward = lstm_out[:, 0, self.hidden_size :]  # (Batch, Hidden)
            last_time_step_feature = torch.cat((out_forward, out_backward), dim=1)
        else:
            # Standard: prendiamo l'ultimo timestep
            # A noi interessa fare una previsione basata su TUTTA la storia.
            # La LSTM accumula informazione man mano che legge.
            # Quindi, l'informazione più completa si trova nell'ULTIMO step temporale.
            # Usiamo lo slicing [:, -1, :] per dire:
            #   :  -> Prendi tutti i campioni del batch (tutti i 64 esempi)
            #   -1 -> Prendi solo l'ULTIMO istante temporale
            #   :  -> Prendi tutte le feature della memoria (tutti i 64 neuroni)
            last_time_step_feature = lstm_out[:, -1, :]
            # Ora la shape è diventata: (Batch_Size, Hidden_Size) -> es. (64, 64)
            # Abbiamo "schiacciato" il tempo.

        # --- PASSO B.1: LAYER NORMALIZATION ---
        # Normalizza le attivazioni per stabilizzare il training.
        # Permette learning rate più alti e migliora la convergenza.
        last_time_step_feature = self.layer_norm(last_time_step_feature)

        # --- PASSO C: PREVISIONE (DECODIFICA) ---
        # Passiamo questo vettore "riassunto" al layer lineare.
        # Lui fa i calcoli matematici (y = Wx + b) per trasformare i 64 numeri di memoria
        # nei 24 numeri che rappresentano la potenza prevista per le prossime 24 ore.
        prediction = self.fc(last_time_step_feature)
        # Shape attuale: (Batch_Size, Horizon) -> es. (64, 24)

        # --- PASSO D: AGGIUSTAMENTO FORMA ---
        # Il tuo Dataset restituisce il target Y con forma (Batch, 24, 1).
        # La Loss Function vuole che la previsione abbia la STESSA forma del target.
        # Quindi aggiungiamo una dimensione finta alla fine con .unsqueeze(-1).
        return prediction.unsqueeze(-1)
        # Shape finale: (Batch_Size, Horizon, 1) -> es. (64, 24, 1)
