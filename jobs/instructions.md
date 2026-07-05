# SLURM jobs: `jobs/utils` scripts

Short guide for interactive GPU work on the cluster using the helpers in `jobs/utils/`. Paths below assume you run commands from the **repository root** (use `cd` there first so `--chdir` and bind mounts match what you expect).

Cluster web portal (Open OnDemand): [https://ondemand.sdumont2nd.lncc.br/](https://ondemand.sdumont2nd.lncc.br/)

---

## 1. Start an interactive GPU session (`run_ict_h100.sh`)

This script calls `salloc` on partition `ict-h100`, requests **1 GPU**, **4 CPUs**, **8 GB RAM**, and **2 hours**, then opens an interactive shell on the compute node via `srun --pty bash`. The job starts in whatever directory is your **current working directory** when you launch the script (`--chdir="$PWD"`).

```bash
bash jobs/utils/run_ict_h100.sh
```

Optional: set a custom job name (shown in `squeue`):

```bash
JOB_NAME=training_v1 bash jobs/utils/run_ict_h100.sh
```

Note the printed `SLURM_JOB_ID`; you need it if you disconnect and want to attach again (see below).

---

## 2. Attach to an existing interactive allocation

If you already have an allocation (same job ID) and want another terminal inside it—for example after SSH dropped—use:

```bash
srun --overlap --jobid=<SLURM_JOB_ID> --pty bash
```

Replace `<SLURM_JOB_ID>` with your actual ID (e.g. `333831`).

---

## 3. Cancel an interactive allocation

To **release the GPU** and end the SLURM job entirely, cancel the allocation by job ID. Run this from the **login node** or from any terminal (including a second shell while your session is still open):

```bash
# List your running jobs
squeue -u "$USER"

# Cancel the interactive allocation
scancel <SLURM_JOB_ID>
```

Example:

```bash
scancel 480223
```

Notes:

- `scancel` stops the whole allocation (all `srun` shells attached to that job ID will lose the session).
- This is different from `kill <PID>` or `pkill -f qdrant`, which only stop a process **inside** the session (e.g. Qdrant), not the GPU reservation.
- If you only `exit` the interactive shell, the allocation may still be held until the time limit expires — use `scancel` when you are done with the GPU.

---

## 4. Containers inside the interactive session

Heavy pulls and image builds are usually done on the **login node** (or as documented by your site), not inside the short interactive shell, to avoid wasting GPU reservation time.

### Pull Qdrant (Singularity)

From the repo root (or any layout where `images/` is where you want the SIF):

```bash
mkdir -p images
singularity pull "$(pwd)/images/qdrant.sif" docker://qdrant/qdrant:latest
```

---

## 5. Run Qdrant (`run_qdrant.sh`)

**Run this only from inside the interactive session** you opened with `run_ict_h100.sh` (or any node where Singularity and the image are available as you intend).

From the directory that should hold persistence and resolve paths correctly:

```bash
bash jobs/utils/run_qdrant.sh
```

Behavior:

- Creates `./qdrant_storage` if needed and binds it to `/qdrant/storage` in the container.
- Expects the image at `./images/qdrant.sif` (relative to **current working directory**).
- Starts Qdrant **in the background**.

Endpoints on **that compute node**:

- REST: `http://localhost:6333`
- gRPC: `localhost:6334`

To reach the API from your laptop you typically need SSH port forwarding or the cluster’s documented approach; `localhost` inside the session refers to the compute node, not your laptop.

### Managing Qdrant processes

Run these on the **same node/session** where Qdrant was started (usually your interactive allocation).

```bash
# List Qdrant-related processes
ps aux | grep qdrant

# Stop every process whose command line matches "qdrant"
pkill -f qdrant

# Stop one process by PID (use the PID from ps output)
kill 1555699
```

Notes:

- The `grep` line often appears in the listing too; that is expected. For a cleaner view you can use `pgrep -af qdrant`.
- If `kill <PID>` does not stop it, try `kill -9 <PID>` only when necessary (forces termination).

---

## Quick reference

| Goal | Command |
|------|---------|
| New interactive H100 session | `bash jobs/utils/run_ict_h100.sh` |
| Named job | `JOB_NAME=my_run bash jobs/utils/run_ict_h100.sh` |
| Second shell in same allocation | `srun --overlap --jobid=<ID> --pty bash` |
| List your SLURM jobs | `squeue -u $USER` |
| Cancel interactive allocation | `scancel <SLURM_JOB_ID>` |
| Pull Qdrant image | `singularity pull "$(pwd)/images/qdrant.sif" docker://qdrant/qdrant:latest` |
| Start Qdrant (inside session) | `bash jobs/utils/run_qdrant.sh` |
| List Qdrant processes | `pgrep -af qdrant` (or `ps aux` + `grep qdrant`; see §4) |
| Stop all Qdrant processes | `pkill -f qdrant` |
| Stop one process by PID | `kill <PID>` |
| Qdrant REST API commands | See [qdrant.md](./qdrant.md) |
