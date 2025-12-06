import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def plot_cv_indices(cv_split_generator, n_samples, n_splits):
    """
    Crea un grafico che mostra i fold della Time Series Cross Validation.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. Definiamo i colori ESATTI una volta per tutte
    # Usiamo la colormap 'coolwarm':
    # 0.1 è un bel Blu (Training), 0.9 è un bel Rosso (Validation)
    cmap = plt.cm.coolwarm
    c_train = cmap(0.1)
    c_val = cmap(0.9)

    # Convertiamo il generatore in lista per poterlo scorrere
    splits = list(cv_split_generator)

    # Iteriamo sui fold
    for ii, (tr, tt) in enumerate(splits):
        # tr e tt sono già gli indici (numeri di riga) che ci servono!

        # --- Disegna Training (Blu) ---
        # Usiamo direttamente 'color=c_train' invece di c=indices + cmap
        ax.scatter(tr, [ii + 0.5] * len(tr), color=c_train, marker="_", lw=10)

        # --- Disegna Validation (Rosso) ---
        ax.scatter(tt, [ii + 0.5] * len(tt), color=c_val, marker="_", lw=10)

    # Formattazione Grafico
    yticklabels = [f"Fold {i + 1}" for i in range(n_splits)]
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Indice Temporale (Ore)",
        ylabel="Iterazione",
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, n_samples],
    )

    ax.set_title(f"Strategia di Split: {n_splits} Fold (Expanding Window)", fontsize=15)

    # Legenda (Ora usa le stesse variabili colore del grafico)
    legend_elements = [
        Patch(facecolor=c_val, label="Validation Set"),
        Patch(facecolor=c_train, label="Training Set"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.show()

    # IMPORTANTE: Dato che abbiamo consumato il generatore trasformandolo in lista,
    # se questa funzione dovesse restituire qualcosa, dovremmo restituire la lista 'splits'.
    # Ma dato che nel tuo codice usi plot_cv_indices solo per visualizzare e poi ricrei
    # lo splitter nel training, va benissimo così.
