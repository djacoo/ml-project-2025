# Codebase Logic Review - Problemi e Suggerimenti

## 🔴 PROBLEMI CRITICI

### 1. **Bug in `scaling.py` - Uso di DataFrame originale invece di copia**
**File**: `src/features/scaling.py`, riga 121

**Problema**: Nel metodo `transform()`, viene usato `X[feature]` invece di `X_scaled[feature]` per la trasformazione.

```python
# Riga 121 - ERRATO
X_scaled[feature] = scaler.transform(
    X[feature].values.reshape(-1, 1)  # ❌ Usa X originale invece di X_scaled
).flatten()
```

**Impatto**: Se `X` viene modificato durante il loop (ad esempio da altri transformer), questo può causare inconsistenze. Anche se non è un bug immediato, viola il principio di non modificare l'input.

**Suggerimento**: Usare `X_scaled[feature]` per coerenza:
```python
X_scaled[feature] = scaler.transform(
    X_scaled[feature].values.reshape(-1, 1)
).flatten()
```

---

### 2. **Outlier statistici rilevati ma NON rimossi**
**File**: `src/data/preprocessing.py`, righe 381-407

**Problema**: Il codice rileva gli outlier statistici (3×IQR) ma non li rimuove effettivamente dal dataframe.

```python
# Riga 396-407
extreme_mask = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
extreme_count = extreme_mask.sum()

if extreme_count > 0:
    # ... salva informazioni ...
    print(f"   - {col:25s}: {extreme_count:,} extreme outliers detected")
    # ❌ MANCA: df_clean = df_clean[~extreme_mask]
```

**Impatto**: Gli outlier statistici vengono rilevati e segnalati, ma rimangono nel dataset. Questo può causare problemi nei modelli ML.

**Suggerimento**: Rimuovere effettivamente gli outlier:
```python
if extreme_count > 0:
    # ... logging ...
    df_clean = df_clean[~extreme_mask]  # ✅ Rimuovi gli outlier
    rows_removed += extreme_count
```

**Nota**: Potrebbe essere intenzionale (solo logging), ma il commento dice "Remove statistical outliers", quindi dovrebbe rimuoverli.

---

### 3. **Potenziale problema di allineamento indici in `run_preprocessing_pipeline.py`**
**File**: `scripts/run_preprocessing_pipeline.py`, riga 77

**Problema**: Se `X_processed` ha meno righe di `X` originale (perché alcune righe sono state rimosse), l'allineamento di `y` potrebbe fallire se gli indici non corrispondono.

```python
# Riga 77
y_aligned = y.loc[X_processed.index]  # ⚠️ Potrebbe fallire se indici non allineati
```

**Impatto**: Se gli indici di `X_processed` non sono un subset degli indici di `y`, questo causerà un `KeyError`.

**Suggerimento**: Aggiungere controllo o usare `reindex`:
```python
# Opzione 1: Verifica
if not X_processed.index.isin(y.index).all():
    raise ValueError("Indices mismatch between X_processed and y")

# Opzione 2: Reindex (più sicuro)
y_aligned = y.reindex(X_processed.index)
if y_aligned.isna().any():
    raise ValueError("Some indices in X_processed not found in y")
```

---

## ⚠️ PROBLEMI LOGICI / INCONSISTENZE

### 4. **Gestione target in `preprocessing_pipeline.py` - Logica complessa e potenzialmente confusa**
**File**: `src/features/preprocessing_pipeline.py`, metodo `fit()`, righe 156-178

**Problema**: La logica per gestire il target durante il fit è complessa e potrebbe causare confusione:

```python
# Riga 157-174
for step_name, transformer in self.pipeline.steps:
    if self.target_col in X_current.columns and y_current is None:
        y_current = pd.Series(X_current[self.target_col].values, index=X_current.index)
    
    if step_name == 'encoding' and y_current is None:
        raise ValueError(...)
    
    transformer.fit(X_current, y_current)
    X_transformed = transformer.transform(X_current)
    
    if len(X_transformed) < len(X_current):
        remaining_indices = X_transformed.index
        if y_current is not None:
            y_current = y_current.loc[remaining_indices]
        if self.target_col in X_transformed.columns and y_current is not None:
            X_transformed[self.target_col] = y_current.values
    
    X_current = X_transformed
```

**Problemi**:
- Il target viene estratto da `X_current` se presente, ma poi viene anche aggiunto a `X_transformed` se presente in `X_transformed.columns`. Questo può creare duplicazioni o confusione.
- La logica di allineamento di `y_current` dopo la rimozione di righe è corretta, ma potrebbe essere più chiara.

**Suggerimento**: Separare meglio la logica del target e documentare meglio il flusso.

---

### 5. **`FeatureEngineer` non è incluso nella pipeline principale**
**File**: `src/features/preprocessing_pipeline.py`

**Problema**: La classe `FeatureEngineer` esiste e crea feature derivate, ma non è inclusa nella pipeline principale `PreprocessingPipeline`.

**Impatto**: Le feature derivate (ratios, energy density, etc.) non vengono create automaticamente nella pipeline.

**Suggerimento**: Aggiungere `FeatureEngineer` come step opzionale nella pipeline, probabilmente dopo l'encoding e prima dello scaling:
```python
steps = [
    ('missing_values', ...),
    ('outlier_removal', ...),
    ('encoding', ...),
    ('feature_engineering', FeatureEngineer()),  # ✅ Aggiungere qui
    ('scaling', ...),
    ('pca', ...)
]
```

---

### 6. **Mancanza di validazione delle colonne richieste**
**File**: Vari file (encoding.py, scaling.py, feature_engineering.py)

**Problema**: Molti transformer assumono che certe colonne esistano, ma non validano all'inizio del `fit()` o `transform()`.

**Esempio in `feature_engineering.py`**:
- `_add_ratios()` controlla se le colonne esistono prima di usarle (✅ buono)
- Ma non c'è validazione generale all'inizio che le colonne necessarie siano presenti

**Suggerimento**: Aggiungere validazione esplicita:
```python
def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
    required_cols = ['fat_100g', 'proteins_100g', ...]  # se necessario
    missing = [col for col in required_cols if col not in X.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return self
```

---

### 7. **Inconsistenza nella gestione di `split_group`**
**File**: `src/features/preprocessing_pipeline.py`

**Problema**: Il parametro `split_group_col` ha un default `'split_group'`, ma se la colonna non esiste, viene semplicemente ignorata senza warning esplicito (solo un print in `run_preprocessing_pipeline.py`).

**Suggerimento**: Aggiungere un warning più esplicito nel metodo `_extract_train_split()`:
```python
def _extract_train_split(self, X: pd.DataFrame, y: Optional[pd.Series]):
    if self.split_group_col not in X.columns:
        import warnings
        warnings.warn(
            f"'{self.split_group_col}' column not found. Fitting on all data. "
            "This may cause data leakage if test data is included.",
            UserWarning
        )
    # ... resto del codice
```

**Status**: ✅ **[RISOLTO]** - Warning esplicito aggiunto nel metodo `_extract_train_split()`.

---

### 8. **PCA: potenziale problema se colonne numeriche cambiano**
**File**: `src/features/dimensionality_reduction.py`

**Problema**: Se tra `fit()` e `transform()` le colonne numeriche cambiano (ad esempio, alcune vengono rimosse), il transform fallirà.

**Suggerimento**: Aggiungere validazione:
```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    if self.pca_ is None:
        raise ValueError("PCA has not been fitted. Call fit() first.")
    
    # Validazione colonne
    missing_cols = [col for col in self.feature_columns_ if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # ... resto del codice
```

**Status**: ✅ **[RISOLTO]** - Validazione già presente nel metodo `transform()` (righe 136-142).

---

## 💡 SUGGERIMENTI DI MIGLIORAMENTO

### 9. **Mancanza di logging strutturato**
**Problema**: Il codice usa principalmente `print()` invece di un sistema di logging strutturato.

**Suggerimento**: Usare il modulo `logging` di Python per:
- Livelli di log (DEBUG, INFO, WARNING, ERROR)
- Possibilità di disabilitare output in produzione
- Logging a file

---

### 10. **Mancanza di type hints completi**
**Problema**: Alcuni metodi non hanno type hints completi per i valori di ritorno o parametri opzionali.

**Esempio**: `preprocessing_pipeline.py` ha buoni type hints, ma alcuni metodi helper potrebbero essere più espliciti.

---

### 12. **Documentazione docstring inconsistente**
**Problema**: Alcuni docstring sono molto dettagliati, altri meno. Alcuni non documentano i possibili errori (Raises section).

**Suggerimento**: Standardizzare i docstring seguendo il formato NumPy/SciPy con sezioni:
- Parameters
- Returns
- Raises
- Notes
- Examples (dove utile)

**Status**: ✅ **[MIGLIORATO]** - Aggiunte sezioni Raises e Notes ai docstring principali:
- `PreprocessingPipeline.fit()` e `transform()`
- `MissingValueTransformer.transform()`
- `OutlierRemovalTransformer.transform()`
- `FeatureEncoder.fit()`

I docstring principali ora seguono il formato NumPy/SciPy standard.

---

## 📋 RIEPILOGO PRIORITÀ

### Alta Priorità (da fixare subito):
1. ✅ **Bug scaling.py riga 121** - Usa X invece di X_scaled **[RISOLTO]**
2. ✅ **Outlier statistici non rimossi** - Logica incompleta **[RISOLTO]**
3. ✅ **Allineamento indici in run_preprocessing_pipeline.py** - Potenziale KeyError **[RISOLTO]**

### Media Priorità (miglioramenti importanti):
4. ✅ **Aggiungere FeatureEngineer alla pipeline** **[RISOLTO]**
5. ✅ **Validazione colonne richieste** **[RISOLTO]**
7. ✅ **Warning per split_group mancante** **[RISOLTO]**
8. ✅ **Validazione PCA colonne** **[RISOLTO]**

### Bassa Priorità (miglioramenti code quality):
7. 💡 **Sistema di logging strutturato**
8. 💡 **Type hints completi**
9. 💡 **Documentazione standardizzata**

---

## ✅ COSE CHE FUNZIONANO BENE

1. ✅ **Gestione data leakage**: Il sistema di `split_group` per fit solo su train è ben implementato
2. ✅ **Preservazione metadata**: Le colonne metadata (product_name, brands, etc.) sono preservate correttamente
3. ✅ **API scikit-learn**: I transformer seguono correttamente l'API scikit-learn
4. ✅ **Modularità**: Il codice è ben organizzato in moduli separati
5. ✅ **Gestione missing values**: La strategia di imputazione è ben definita

---

*Review completata il: 2026-01-25*
