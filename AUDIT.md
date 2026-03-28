# Code audit — `supra_hodge_reasoning_pipeline.py`

**Date:** 2026-03-28 (audit pass)

## Scope

Single-module reference implementation: dense linear algebra, Jacobi eigensolver, simplicial / Hodge / Supra-Hodge assembly, mock LLM pipeline, `.env` loading, OpenAI HTTP client (stdlib).

## Security

| Topic | Assessment |
|--------|----------------|
| API key handling | Key read from environment or `.env`; not logged. `.env` is gitignored. Authorization uses `Bearer` header only in transit to `api.openai.com`. |
| Deserialization | JSON responses parsed with `json.loads`; no `eval`. |
| Local file read | `.env` read as UTF-8 text; no path traversal beyond chosen candidate paths. |

## Correctness fixes applied (this audit)

1. **`build_eight_node_case_study` return type** — Annotated return now includes the third value `edge_list` (`Tuple[..., ..., List[Tuple[int, int]]]`).
2. **`block_matrix`** — Replaced `blocks.index(row)` (fragile / quadratic) with `enumerate(blocks)`.
3. **`mat_mul` / `mat_vec_mul` / `vec_dot` / `vec_sub`** — Explicit dimension checks with `ValueError` instead of silent truncation.
4. **`openai_chat_completion`** — Avoids bare `KeyError` on malformed API JSON; returns a short diagnostic string if `choices` / `message` / `content` is missing.
5. **`spectral_projector_apply`** — Safe behavior when `eigenvectors` is empty.
6. **`jacobi_symmetric_eigen`** — (a) Convergence warning if `max_sweeps` is exhausted with residual off-diagonal mass. (b) **Root cause of non-convergence:** skipping rotations when `|a_pq| < tol` used the same `tol` as the global Frobenius stop; many moderate entries could leave the global norm above `tol` forever. Skips now use a scale-aware near-zero threshold instead. (c) Default `max_sweeps` set to 150 after the fix (sufficient for current demo matrices).

## Known limitations (by design)

| Limitation | Notes |
|------------|--------|
| Dense matrices only | Complexity \(O(n^3)\) for Jacobi; fine for demo size, not for large sparse production graphs. |
| Jacobi vs LAPACK | No shift-invert or Lanczos; for ill-conditioned or very large systems, prefer a production eigensolver. |
| `mat_add` / `mat_sub`** | Still assume matching shapes; callers are internal. Could add checks if exposed as a library. |
| HTTP errors | Rate limits / billing (e.g. HTTP 429) return text to the caller; not retried with backoff. |
| `load_env_file` | Values are single-line `KEY=VALUE`; leading/trailing quotes stripped; inline `=` in values are not supported (rare for API keys). |

## Suggested follow-ups (optional)

- Add `pytest` tests: small Laplacian eigenvalues vs known spectra; energy identity (9); `2×2` Jacobi ground truth.
- For OpenAI: optional retry on 429 with exponential backoff; redact error bodies in logs if logging is added later.
- Consider exporting only a thin public API (`__all__`) if the file is imported as a package.

## Regression check

- `python supra_hodge_reasoning_pipeline.py` completes; energy check \(\|v^\top L v - \text{(9)}\|\) remains \(\sim 10^{-14}\).
- `warnings.simplefilter('error', RuntimeWarning)` on `main()` passes (no Jacobi convergence warning on default demo).
