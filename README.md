# Confronto tra RAG e LLM Wiki

Il benchmark `structured_wiki_v4` confronta sei sistemi di domanda e risposta su MultiHop-RAG. L'archivio conserva le quattro esecuzioni finali del 16 novembre 2026. Qwen2.5-3B-Instruct e Qwen3.5-4B hanno risposto, ciascuno, a 25 domande su 3 documenti e a 150 domande su 8 documenti. Tutti e sei i sistemi hanno completato ogni condizione, per un totale di **2.100 risposte valutate**.

## Ripristinare dati e risultati

La copia amministrativa dello ZIP va conservata sulla **macchina 40**, nella stessa home che contiene la repository, accanto alla directory del clone. Per lavorare, copiare lo ZIP accanto a un clone pulito ed estrarlo **dalla radice del clone**. Il contenuto dello ZIP parte direttamente da `data/` e `results/`.

```bash
git clone --depth 1 git@github.com:burraco04/llm-wiki-comparisons.git
cd llm-wiki-comparisons
archive=../llm-wiki-v4.zip
python3 tools/archive_checkpoints.py "$archive" --verify
python3 -m zipfile -e "$archive" .
```

Lo ZIP dei dati occupa 61.808.856 byte. Il suo SHA-256 è `08765fa7bf9ddc1c64204d8e7722926df51ccc4a29e26d1c2a59d2ed292bfa57`. Per la copia che include anche i pesi, impostare `archive=../llm-wiki-v4-completo.zip` prima degli stessi comandi. Il verificatore controlla i singoli file rispetto ai manifest dell'archivio.

I vecchi commit contengono risultati e log che all'epoca erano tracciati da Git. Il clone con `--depth 1` scarica l'ultima revisione. Su un vecchio checkout del server, un `git pull` che applica la pulizia rimuove i file prima tracciati: verificare la copia amministrativa prima di aggiornare quel checkout.

| Percorso dopo l'estrazione | Contenuto |
| --- | --- |
| `data/multihoprag/` | Corpus originale di 609 articoli, 2.556 domande e scheda del dataset. Revisione Hugging Face `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`. |
| `results/structured_wiki_v4/<modello>/docs_<N>_questions_<M>/<timestamp>/dataset/` | Documenti e domande effettivamente usati in quel run, nell'ordine originale. |
| Directory padre di `dataset/` | Manifest, versioni software, prompt, wiki costruite, indici, checkpoint delle risposte, tracce, metriche e tempi. |
| `checkpoints/huggingface/hub/` | Snapshot dei quattro modelli nella copia dello ZIP completata sul server. |

## Verificare i risultati su CPU

Dopo l'estrazione, questi comandi usano la libreria standard Python. Il primo ricostruisce i due sottoinsiemi dal dataset originale e li confronta con i quattro manifest. Il secondo ricalcola le statistiche dalle risposte registrate.

```bash
python3 analysis/audit_dataset.py
python3 analysis/audit_results.py
```

I report vengono scritti sotto `results/analysis/`. Per ispezionare una singola esecuzione, aprire `run_status.json` e `summary.csv` nella directory del run. In `full_results.json` sono conservati anche i tre risultati di componente, alle chiavi `structured_wiki`, `llmwiki2.classic` e `llmwiki2.crow`. Nell'archivio sono state eliminate le dodici copie ridondanti di questi dizionari. I dettagli delle metriche sono nella [guida all'analisi](analysis/README.md).

## Eseguire il benchmark

Su Linux con Docker, due GPU NVIDIA disponibili per job e Slurm, eseguire dalla radice della repository:

```bash
docker build -t llm-wiki-benchmark:structured-wiki-v4 .
python3 verify_benchmark_image.py llm-wiki-benchmark:structured-wiki-v4
REPLAY_ARCHIVE=1 bash run_both_models.sh
```

Il launcher confronta i file dell'immagine con il checkout, avvia generatore e classificatore e scrive i nuovi risultati in directory con nuovi timestamp. Il comando usa i sottoinsiemi già estratti. Per ricalcolarli dal dataset remoto alla revisione registrata, eseguire `bash run_both_models.sh`. Le istruzioni per scegliere il nodo Slurm, avviare una singola condizione senza Slurm, usare i pesi offline e lanciare i test sono nella [guida di esecuzione](docs/runtime.md).

| Sistema | Come seleziona le evidenze |
| --- | --- |
| `rag_semantic` | Recupera i tre segmenti di testo sorgente più vicini alla domanda per similarità degli embedding. |
| `wiki_routed` | Il modello seleziona pagine e termini di ricerca. Il sistema ordina i paragrafi per parole chiave e include relazioni del grafo. |
| `wiki_bm25` | Esegue la stessa preparazione della domanda e ordina con BM25 i paragrafi della wiki e delle fonti. |
| `wiki_hybrid` | Combina con Reciprocal Rank Fusion le classifiche BM25 e dense. |
| `llmwiki2_classic` | Costruisce una wiki di note e usa il modello generativo per le decisioni di organizzazione e recupero. |
| `llmwiki2_crow` | Costruisce una seconda wiki di note e usa il Decision Model, con fallback generativo. |

Le tre strategie `wiki_*` condividono una wiki strutturata. Classic e CROW costruiscono ciascuno la propria wiki con documenti e prompt nello stesso ordine. La versione v4 registra sia i tentativi di recupero da JSON troncato sia il loro costo in chiamate, token e tempo. Le risposte prodotte durante la valutazione non vengono reinserite nelle wiki.

## Aggiungere allo ZIP i pesi della macchina sperimentale

I quattro modelli sono i due Qwen, il classificatore `LiquidAI/d1-3B` e l'encoder `sentence-transformers/all-MiniLM-L6-v2`. Gli esperimenti hanno utilizzato checkpoint preaddestrati. I manifest registrano gli identificativi dei modelli, ma non i rispettivi commit. Per conservare le revisioni effettivamente disponibili sulla macchina, eseguire dalla radice del clone con la cache Hugging Face usata dagli esperimenti:

```bash
python3 tools/archive_checkpoints.py ../llm-wiki-v4.zip \
  --cache "${HF_CACHE:-$HOME/.cache/huggingface}" \
  --output ../llm-wiki-v4-completo.zip
python3 tools/archive_checkpoints.py ../llm-wiki-v4-completo.zip --verify
```

Il comando legge la cache esistente, valida gli shard e aggiunge i pesi, il tokenizer e le configurazioni di ciascun modello a un nuovo ZIP. Se per un modello esistono più snapshot, specificare `--revision 'organizzazione/modello=SHA'` con il commit di 40 caratteri della copia usata per l'esperimento. Il risultato include dataset, risultati e checkpoint in **un solo archivio**. La verifica finale legge tutti i file e ne controlla gli SHA-256. Quel file è la copia da conservare accanto alla repository sulla 40.
