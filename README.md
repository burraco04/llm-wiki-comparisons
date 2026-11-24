# Confronto tra RAG e LLM Wiki

Il benchmark `structured_wiki_v4` confronta sei sistemi di domanda e risposta su MultiHop-RAG usando due modelli, Qwen2.5-3B-Instruct e Qwen3.5-4B.

## Ripristinare i dati

Dalla radice della repository, estrai lo ZIP indicando il percorso del file. Il punto finale indica la cartella corrente.

```bash
python3 -m zipfile -e ../llm-wiki-v4.zip .
```

Troverai il dataset in `data/multihoprag/` e i risultati in `results/structured_wiki_v4/`. Se hai l’archivio completo, usa `llm-wiki-v4-completo.zip` nello stesso comando. I pesi verranno estratti in `checkpoints/huggingface/`.

## Eseguire il benchmark

Su Linux con Docker, Slurm e due GPU NVIDIA per job, esegui dalla radice della repository:

```bash
docker build -t llm-wiki-benchmark:structured-wiki-v4 .
RUN_NODE='' REPLAY_ARCHIVE=1 bash run_both_models.sh
```

Slurm sceglie il nodo e avvia quattro esperimenti. Ogni modello risponde a 25 domande su 3 documenti e a 150 domande su 8 documenti. Trovi le metriche nel file `summary.csv` di ciascuna esecuzione, sotto `results/structured_wiki_v4/`.

Per usare i pesi estratti o eseguire il benchmark senza Slurm, consulta la [guida di esecuzione](docs/runtime.md).
