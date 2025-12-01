import pandas as pd
import torch
from transformers import PatchTSTForPrediction, PatchTSTConfig

# ----------------------------------------------------------
# 1. CONFIGURAZIONE E PREPARAZIONE DATI
# ----------------------------------------------------------
print("--- INIZIO PROCESSO ---")
model_name = "ibm-granite/granite-timeseries-patchtst"


# A. Carichiamo i dati PRIMA del modello per sapere quanti canali ci servono
try:
    df = pd.read_csv('Data/dataset_ready_for_granite.csv')
    print("Dataset caricato correttamente.")
except FileNotFoundError:
    print("ATTENZIONE: File non trovato in 'Data/'. Provo nella cartella corrente...")
    df = pd.read_csv('dataset_ready_for_granite.csv')

# Calcoliamo i canali reali (dovrebbero essere 13 nel tuo caso)
num_tuoi_canali = df.shape[1] 
print(f"Canali rilevati nel dataset: {num_tuoi_canali}")

# ----------------------------------------------------------
# 2. CARICAMENTO MODELLO CON FORZATURA CANALI (IL FIX REALE)
# ----------------------------------------------------------

# A. Scarichiamo solo la configurazione
config = PatchTSTConfig.from_pretrained(model_name)

# B. Modifichiamo la configurazione 'a tavolino'
config.num_input_channels = num_tuoi_canali

# C. Carichiamo il modello passandogli la configurazione modificata
# 'ignore_mismatched_sizes=True' è fondamentale: dice a PyTorch di non andare in panico
# se i pesi scaricati (per 7 canali) non combaciano perfettamente con la nuova struttura (13 canali).
# Grazie alla Channel Independence, i pesi fondamentali verranno riutilizzati correttamente.
print(f"Caricamento modello forzato a {num_tuoi_canali} canali...")
model = PatchTSTForPrediction.from_pretrained(
    model_name, 
    config=config, 
    ignore_mismatched_sizes=True
)

print("Modello costruito e caricato con successo.")
model.eval()

# ----------------------------------------------------------
# 3. PREPARAZIONE TENSORI
# ----------------------------------------------------------
context_length = config.context_length

if len(df) < context_length:
    raise ValueError(f"Il dataset ha {len(df)} righe, ma il modello ne richiede almeno {context_length}.")

# Prendiamo l'ultima finestra
input_data = df.tail(context_length)

# Creazione tensore [Batch, Sequence, Channels]
tensor_values = torch.tensor(input_data.values).float()
batch_input = tensor_values.unsqueeze(0) 

# Verifica dimensionale finale prima dell'inferenza
print(f"Shape Input: {batch_input.shape}") # Deve essere [1, 512, 13]
print(f"Canali attesi dal modello: {model.config.num_input_channels}") # Deve essere 13

# ----------------------------------------------------------
# 4. ESECUZIONE INFERENZA
# ----------------------------------------------------------
print("Avvio inferenza...")
with torch.no_grad():
    outputs = model(past_values=batch_input)

predictions = outputs.prediction_outputs
preds_numpy = predictions.squeeze(0).numpy()

# ----------------------------------------------------------
# 5. RISULTATI
# ----------------------------------------------------------
print("\n--- TEST COMPLETATO CON SUCCESSO ---")
print(f"Output generato: {preds_numpy.shape} (Step Futuri x Numero Variabili)")

# Visualizzazione (PV Power è l'ultima colonna)
last_col_idx = -1 
last_col_name = df.columns[last_col_idx]

print(f"\nPrimi 5 valori predetti per la colonna '{last_col_name}':")
print(preds_numpy[:5, last_col_idx])