# Riprodurre il benchmark finale

I comandi si eseguono dalla radice della repository su Linux. I quattro job
del 16 novembre 2026 hanno utilizzato due NVIDIA RTX 3090 da 24 GiB per job,
driver 570.211.01, quattro CPU e 16 GiB di RAM richiesti a Slurm. Occorrono
Bash, Python 3 sull’host, curl, nvidia-smi e Docker con supporto GPU NVIDIA.
La macchina di esecuzione deve poter leggere il checkout e usare l’immagine
Docker costruita. Su un cluster, sia il nodo di invio sia quelli scelti da
Slurm devono poter verificare la stessa immagine.

## Costruire e controllare l’immagine

```bash
docker build -t llm-wiki-benchmark:structured-wiki-v4 .
python3 verify_benchmark_image.py llm-wiki-benchmark:structured-wiki-v4
```

Il Dockerfile usa `vllm/vllm-openai:v0.32.0-cu129`. Le versioni delle
dipendenze dirette in `requirements.txt` provengono dai quattro file
`runtime_environment.json` finali. Per costruire l’immagine servono accesso
al registro Docker e agli indici dei pacchetti, oppure copie locali delle
risorse necessarie. Il controllo stampa l’ID dell’immagine locale verificata.

I test si eseguono dalla stessa directory con i pacchetti dell’immagine.

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/checkout" -w /checkout --entrypoint python \
  llm-wiki-benchmark:structured-wiki-v4 -m unittest discover -v
```

## Ripetere i quattro job con i documenti e le domande originali

Estrarre prima lo ZIP come indicato nel README. L’opzione `REPLAY_ARCHIVE=1`
legge i due sottoinsiemi ripristinati sotto `results/structured_wiki_v4/`.
Le due esecuzioni a pari dimensione hanno usato identici documenti e domande,
nello stesso ordine. Il benchmark verifica ordine, titoli, hash SHA-256 dei
testi e delle domande rispetto al manifest originale prima di chiamare i modelli.
Ogni nuova esecuzione scrive in una nuova directory con un proprio timestamp.

```bash
REPLAY_ARCHIVE=1 bash run_both_models.sh
```

Il launcher usa il nodo storico `faretra`. Impostare `RUN_NODE` al nome del
proprio nodo o passare `RUN_NODE='' REPLAY_ARCHIVE=1 bash run_both_models.sh`
per lasciare la scelta a Slurm. I job usano Qwen2.5-3B-Instruct e Qwen3.5-4B,
ciascuno su 3 documenti e 25 domande e su 8 documenti e 150 domande, con
sei configurazioni per esecuzione. `RUN_TIME_LIMIT` controlla il tempo di
allocazione e vale otto ore per impostazione predefinita.

Su un host dedicato senza Slurm si può eseguire una singola condizione con
lo stesso launcher. Nell’esempio le GPU disponibili hanno indici 0 e 1.

```bash
small=/workdir/results/structured_wiki_v4/Qwen_Qwen2.5-3B-Instruct/docs_3_questions_25/20261116T110157_135698Z/dataset
CUDA_VISIBLE_DEVICES=0,1 bash run_with_vllm.sh Qwen/Qwen2.5-3B-Instruct -- \
  python benchmark_llm_wiki.py --top-docs 3 --max-questions 25 --dataset-dir "$small"
```

Per ricostruire i sottoinsiemi da Hugging Face, eseguire
`bash run_both_models.sh`. Il benchmark seleziona i record di
`yixuantt/MultiHopRAG` alla revisione
`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82` registrata nei quattro
run. Una singola invocazione accetta anche `--dataset-revision <commit>`.

## Pesi e cache dei modelli

I due generatori usano `Qwen/Qwen2.5-3B-Instruct` e `Qwen/Qwen3.5-4B`.
Le altre due risorse sono il classificatore `LiquidAI/d1-3B` e l’encoder
`sentence-transformers/all-MiniLM-L6-v2`. Il launcher monta la cache host
`${HF_CACHE:-$HOME/.cache/huggingface}` in `/root/.cache/huggingface`.
Al primo avvio serve accesso a Hugging Face per i modelli assenti.

Se lo ZIP include le quattro copie della cache del server, questo comando
usa la cache ripristinata in modalità offline.

```bash
HF_CACHE="$PWD/checkpoints/huggingface" HF_HUB_OFFLINE=1 \
  REPLAY_ARCHIVE=1 bash run_both_models.sh
```

La cache deve contenere i quattro alberi
`hub/models--<organizzazione>--<modello>/` con `refs/main` e lo snapshot
corrispondente completo, inclusi pesi, tokenizer e configurazione. I file a
cui puntano eventuali link in `snapshots/` devono essere leggibili dopo
l’estrazione. I log locali riportano `main` e i manifest finali riportano i
nomi dei modelli. Gli SHA storici dei quattro checkpoint non sono documentati.
Per una verifica degli stessi pesi occorrono le copie originali del server e
la loro provenienza. Un download odierno può risolvere una revisione successiva.

## Leggere il risultato

Per ciascuna combinazione, aprire la nuova directory
`results/structured_wiki_v4/<modello>/docs_<N>_questions_<M>/<timestamp>/`.
In `run_status.json`, `comparison_valid` indica se il confronto richiesto è
stato completato. `summary.csv` contiene le metriche per sistema e
`paired_comparison.json` riporta il confronto classic–CROW. Il manifest
contiene gli hash degli input e la configurazione. `runtime_environment.json`
registra hardware, versioni e comandi dei servizi usati in quel run.
