import pandas as pd
import numpy as np
from features_extractor import extract_all_abc_features
from util.Data_Loader import load_and_preprocess_data, show_image_and_mask_examples

df_filtered = load_and_preprocess_data()
# show_image_and_mask_examples(df_filtered) # Uncomment to see images


print("--- Inizio del pipeline di analisi ---")

# 1. Carica il DataFrame pre-elaborato (df_filtered)
# Questa funzione gestirà tutto il caricamento dei metadati, la verifica dei file,
# e il caricamento delle immagini/maschere in memoria.
df_filtered = load_and_preprocess_data()

# Controlla se il caricamento dei dati è avvenuto con successo
if df_filtered is None or df_filtered.empty:
    print("Errore: Impossibile caricare il DataFrame. Nessun dato valido per l'estrazione delle feature.")
    # Esci dallo script se non ci sono dati validi
    exit() 
else:
    print(f"\nDataFrame pre-elaborato caricato con successo. Righe totali: {len(df_filtered)}")
    # OPTIONAL: Se vuoi visualizzare alcuni esempi di immagini e maschere dopo il caricamento,
    # decommenta la riga seguente.
    # show_image_and_mask_examples(df_filtered, num_examples=3)

# 2. Estrazione delle Feature ABC per tutte le immagini
print("\nInizio estrazione delle feature ABC per tutti i campioni...")
abc_features_list = []

# Itera su ogni riga del df_filtered per accedere a immagine e maschera
for index, row in df_filtered.iterrows():
    img = row['image']
    mask = row['mask']

    # Esegui l'estrazione delle feature solo se immagine e maschera sono state caricate correttamente
    # (df_filtered.dropna() nel Data_Loader dovrebbe già aver rimosso i casi problematici,
    # ma un controllo qui aggiunge robustezza)
    if img is not None and mask is not None:
        try:
            current_features = extract_all_abc_features(img, mask)
            abc_features_list.append(current_features)
        except Exception as e:
            # Cattura errori specifici dell'estrazione feature per una riga
            print(f"Attenzione: Errore durante l'estrazione feature per ID {row.get('img_id', 'N/A')} (indice {index}): {e}")
            # Aggiungi NaN per le feature se l'estrazione fallisce per questa riga
            abc_features_list.append({
                'rotational_asymmetry_score': np.nan,
                'compactness_score': np.nan,
                'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
                'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan
            })
    else:
        # Questo caso dovrebbe essere raro se df_filtered.dropna() funziona correttamente,
        # ma è una rete di sicurezza.
        print(f"Attenzione: Immagine o maschera mancante per ID {row.get('img_id', 'N/A')} (indice {index}). Aggiungo feature NaN.")
        abc_features_list.append({
            'rotational_asymmetry_score': np.nan,
            'compactness_score': np.nan,
            'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
            'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan
        })

print("\nEstrazione delle feature ABC completata.")

# 3. Converti la lista di dizionari di feature in un DataFrame separato
abc_features_df = pd.DataFrame(abc_features_list)

# 4. Concatena le nuove feature al tuo DataFrame esistente (df_filtered)
# Usiamo reset_index(drop=True) su entrambi per assicurarci che gli indici si allineino correttamente
df_final = pd.concat([df_filtered.reset_index(drop=True), abc_features_df.reset_index(drop=True)], axis=1)

print("\nDataFrame finale con le feature ABC (prime 5 righe e colonne delle feature selezionate):")
# Seleziona solo le colonne delle feature ABC per una visualizzazione più chiara
feature_cols_to_display = [col for col in df_final.columns if 'asymmetry' in col or 'compactness' in col or 'color' in col]
# Aggiungi anche alcune colonne di metadati per contesto
context_cols = ['img_id', 'diagnostic', 'label']
# Filtra per le colonne che esistono effettivamente nel df_final
final_display_cols = [col for col in context_cols + feature_cols_to_display if col in df_final.columns]
print(df_final[final_display_cols].head())

# 5. Rimuovi le colonne 'image' e 'mask'
# Queste colonne contengono gli array NumPy delle immagini, che sono pesanti e non servono più
# una volta estratte le feature numeriche per il modello ML.
if 'image' in df_final.columns:
    df_final = df_final.drop(columns=['image', 'mask'])
    print("\nColonne 'image' e 'mask' rimosse dal DataFrame finale.")

print("\nPipeline di estrazione feature completata. Il DataFrame 'df_final' è pronto per l'addestramento del modello ML.")
   