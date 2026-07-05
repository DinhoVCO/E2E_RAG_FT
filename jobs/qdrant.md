# Qdrant on Santos Dumont

Quick reference for running Qdrant inside an interactive GPU session and inspecting collections via the REST API.

Run these commands from the **repository root** unless noted otherwise. `localhost` refers to the **compute node** where your session is running, not your laptop.

See also: [instructions.md](./instructions.md) for SLURM session setup.

---

## Start and stop

### Pull the Singularity image (login node)

```bash
mkdir -p images
singularity pull "$(pwd)/images/qdrant.sif" docker://qdrant/qdrant:latest
```

### Start Qdrant (inside the interactive session)

```bash
bash jobs/scripts/santos_dumont/run_qdrant.sh
```

- Persists data under `./qdrant_storage`
- REST API: `http://localhost:6333`
- gRPC: `localhost:6334`

### Check that Qdrant is running

```bash
pgrep -af qdrant
curl -s http://localhost:6333/ | jq
```

### Stop Qdrant

```bash
pkill -f qdrant
```

---

## REST API (curl)

Set the base URL once (optional):

```bash
export QDRANT_URL=http://localhost:6333
```

### List all collections

```bash
curl -s "$QDRANT_URL/collections" | jq
```

Without `jq`:

```bash
curl -s "$QDRANT_URL/collections"
```

### Collection details (vector size, point count, status)

```bash
curl -s "$QDRANT_URL/collections/qasper-rag-corpus" | jq
```

### Point count only

```bash
curl -s "$QDRANT_URL/collections/qasper-rag-corpus" | jq '.result.points_count'
```

### Scroll a few points (payload preview)

```bash
curl -s "$QDRANT_URL/collections/qasper-rag-corpus/points/scroll" \
  -H 'Content-Type: application/json' \
  -d '{"limit": 3, "with_payload": true, "with_vector": false}' | jq
```

### Delete a collection

```bash
curl -s -X DELETE "$QDRANT_URL/collections/qasper-rag-corpus" | jq
```

---

## Default indexing collections

These names match the defaults in `scripts/embeddings/index_*_corpus.py`:

| Collection | Dataset | Index script |
|------------|---------|--------------|
| `bioasq-rag-13b-corpus` | `dinho1597/bioasq-rag-13b` | `scripts/embeddings/index_bioasq_corpus.py` |
| `qasper-rag-corpus` | `dinho1597/qasper-rag` | `scripts/embeddings/index_qasper_corpus.py` |
| `telco-dpr-rag-corpus` | `dinho1597/telco-dpr-rag` | `scripts/embeddings/index_telco_dpr_corpus.py` |
| `narrativeqa-rag-corpus` | `dinho1597/narrativeqa-rag` | `scripts/embeddings/index_narrativeqa_corpus.py` |

Example: check whether QASPER indexing finished:

```bash
curl -s "$QDRANT_URL/collections/qasper-rag-corpus" | jq '.result | {status, points_count, indexed_vectors_count}'
```

Expected point counts (approximate):

| Collection | Corpus size |
|------------|------------:|
| `bioasq-rag-13b-corpus` | varies |
| `qasper-rag-corpus` | ~81,550 |
| `telco-dpr-rag-corpus` | ~14,654 |
| `narrativeqa-rag-corpus` | ~1,572 |

---

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| `Connection refused` on port 6333 | Qdrant is not running — start it with `run_qdrant.sh` |
| Client/server version warning | Usually harmless for indexing; upgrade `qdrant-client` when convenient |
| Empty collection after a failed run | Re-run the index script; use `--recreate-collection` to start fresh |
| Cannot reach API from laptop | SSH port-forward: `ssh -L 6333:localhost:6333 <user>@<login-node>` while attached to the compute session |

---

## Quick reference

| Goal | Command |
|------|---------|
| Start Qdrant | `bash jobs/scripts/santos_dumont/run_qdrant.sh` |
| Health check | `curl -s http://localhost:6333/` |
| List collections | `curl -s http://localhost:6333/collections \| jq` |
| Collection info | `curl -s http://localhost:6333/collections/<name> \| jq` |
| Point count | `curl -s http://localhost:6333/collections/<name> \| jq '.result.points_count'` |
| List processes | `pgrep -af qdrant` |
| Stop Qdrant | `pkill -f qdrant` |
