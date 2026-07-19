from tesis_unicamp.query_expansion.hyde.embedding import compute_hyde_vector, compute_hyde_vectors_batch
from tesis_unicamp.query_expansion.hyde.evaluation import (
    HYDE_DATASET_IDS,
    HydeMtebEvalConfig,
    evaluate_hyde_mteb,
    resolve_paper_scoped,
)
from tesis_unicamp.query_expansion.hyde.experiment_config import (
    ResolvedHydeExperiment,
    default_experiments_path,
    hyde_output_dir,
    hyde_passages_path,
    list_experiment_ids,
    load_experiments_yaml,
    resolve_experiment,
    resolve_experiments,
)
from tesis_unicamp.query_expansion.hyde.generation import (
    DEFAULT_HYDE_MAX_TOKENS,
    DEFAULT_HYDE_TEMPERATURE,
    DEFAULT_NUM_PASSAGES,
    generate_hyde_passages_for_split,
)
from tesis_unicamp.query_expansion.hyde.io import load_hyde_records, save_hyde_bundle
from tesis_unicamp.query_expansion.hyde.mteb_encoder import (
    HydeMtebEncoder,
    build_instruct_passage_lookup,
)
from tesis_unicamp.query_expansion.hyde.prompts import RAG_HYDE_PROMPT_TEMPLATE, build_hyde_prompt
from tesis_unicamp.query_expansion.hyde.schemas import HydeRecord

__all__ = [
    "DEFAULT_HYDE_MAX_TOKENS",
    "DEFAULT_HYDE_TEMPERATURE",
    "DEFAULT_NUM_PASSAGES",
    "HYDE_DATASET_IDS",
    "HydeMtebEvalConfig",
    "HydeMtebEncoder",
    "HydeRecord",
    "RAG_HYDE_PROMPT_TEMPLATE",
    "ResolvedHydeExperiment",
    "build_hyde_prompt",
    "build_instruct_passage_lookup",
    "compute_hyde_vector",
    "compute_hyde_vectors_batch",
    "default_experiments_path",
    "evaluate_hyde_mteb",
    "generate_hyde_passages_for_split",
    "hyde_output_dir",
    "hyde_passages_path",
    "list_experiment_ids",
    "load_experiments_yaml",
    "load_hyde_records",
    "resolve_experiment",
    "resolve_experiments",
    "resolve_paper_scoped",
    "save_hyde_bundle",
]
