import torch
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, val_loader, device, scaler=None, target_idx=None):
    """
    Esegue predizioni su tutto il validation set, denormalizza e plotta.
    
    Args:
        target_idx (int): L'indice della colonna target (pv_power) nel dataset originale.
                          Serve per pescare il fattore di scala giusto.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Avvio valutazione finale...")
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            
            # 1. Predizione
            outputs = model(batch_x)
            
            # 2. Converti in Numpy
            preds = outputs.cpu().numpy()
            targets = batch_y.numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
            
    # Concateniamo tutto
    all_preds = np.concatenate(all_preds, axis=0)     # Shape: (N, 24)
    all_targets = np.concatenate(all_targets, axis=0) # Shape: (N, 24)
    
    # --- 3. DENORMALIZZAZIONE INTELLIGENTE ---
    if scaler is not None and target_idx is not None:
        print(f"Denormalizzazione in corso usando la colonna idx: {target_idx}...")
        
        # Recuperiamo i parametri SOLO per la colonna target
        # Nota: MinMaxScaler salva i parametri in .scale_ e .min_
        scale_factor = scaler.scale_[target_idx]
        min_factor = scaler.min_[target_idx]
        
        # Formula inversa del MinMaxScaler: X_reale = (X_scaled - min) / scale
        all_preds = (all_preds - min_factor) / scale_factor
        all_targets = (all_targets - min_factor) / scale_factor
        
        print("Output convertito in unità reali (Watt).")
        
    elif scaler is not None and target_idx is None:
        print("⚠️ ATTENZIONE: Scaler fornito ma 'target_idx' mancante. Impossibile denormalizzare correttamente.")

    '''# --- 4. CALCOLO METRICHE REALI ---
    # Calcoliamo il MAE (Mean Absolute Error) in Watt
    mae = np.mean(np.abs(all_preds - all_targets))
    print(f"MAE sul Validation Set: {mae:.2f} Watt")

    # --- PLOTTING ---
    indices_to_plot = [0, 50, 100, 200]
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices_to_plot):
        if idx >= len(all_preds): break
        
        plt.subplot(1, 4, i+1)
        plt.plot(all_targets[idx], label='Reale', color='blue', marker='.')
        plt.plot(all_preds[idx], label='Predetto', color='red', linestyle='--')
        
        plt.title(f"Campione {idx}")
        plt.legend()
        plt.grid(True)
        # Non forziamo più ylim(0,1) perché ora siamo in Watt (es. 0 a 3000)
    
    plt.tight_layout()
    plt.show()'''
    
    return all_preds, all_targets