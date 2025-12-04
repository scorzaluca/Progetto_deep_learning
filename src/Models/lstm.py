import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, model_config: dict, train_loader):
        super(LSTM, self).__init__()
        
        # --- MODIFICA: Lettura dinamica delle features dal loader ---
        # Accede al dataset dentro il loader e legge la shape del tensore dati
        self.input_size = train_loader.dataset.data_tensor.shape[1]
        
        # Il resto delle configurazioni rimane uguale
        self.hidden_size = model_config.get("hidden_size", 64)
        self.output_size = model_config.get("output_size", 24)
        self.num_layers = model_config.get("num_layers", 1)
        self.dropout = model_config.get("dropout", 0.0)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=model_config.get("bidirectional", False),
            batch_first=model_config.get("batch_first", True),
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: è il batch di input.
           Shape attesa: (Batch_Size, Lookback, Input_Size) -> es. (64, 48, 10)
        """

        # --- PASSO A: ELABORAZIONE TEMPORALE ---
        # Passiamo tutto il "filmato" (x) dentro la LSTM.
        # lstm_out: Contiene lo stato della memoria per OGNI istante temporale (t-48, t-47... t-1).
        #           Shape: (Batch_Size, Lookback, Hidden_Size) -> es. (64, 48, 64)
        # _ : Sono gli stati nascosti finali (h_n, c_n). Non ci servono direttamente perché sono già contenuti in lstm_out.
        lstm_out, _ = self.lstm(x)

        # --- PASSO B: SELEZIONE DELL'ULTIMO ISTANTE ---
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