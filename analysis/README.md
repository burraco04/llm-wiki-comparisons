# Verifica dei quattro run v4

Dopo aver estratto lo ZIP nella radice della repository, eseguire dalla stessa directory i due comandi. Basta Python 3, con la libreria standard.

```bash
python3 analysis/audit_dataset.py
python3 analysis/audit_results.py
```

| File prodotto in `results/analysis/` | Verifica eseguita |
| --- | --- |
| `dataset_statistics.json` | Ricostruisce i sottoinsiemi 3/25 e 8/150 dai 609 articoli e dalle 2.556 domande del dataset completo. Confronta titoli, hash dei testi, domande e ordine con i quattro manifest. |
| `results_v4_audit.json` | Verifica lo stato e gli identificativi delle domande per ciascuno dei sei sistemi. Ricalcola dalle singole risposte le medie di EM, F1, ROUGE-L e latenza, le confronta con i riepiloghi dei run e controlla il totale di 2.100 risposte valutate. |

Il secondo report include anche mediana, 95º percentile e massimo della latenza, conteggi per tipo di domanda e costi di costruzione. Per ogni coppia di sistemi, i quattro conteggi in `paired` rappresentano, nell'ordine, risposte esatte per entrambi, per il primo, per il secondo e per nessuno dei due.

Gli script leggono i quattro timestamp originali del 16 novembre 2026, così le esecuzioni successive possono convivere nella stessa directory. Usare `--help` per indicare percorsi di input o di output diversi.
