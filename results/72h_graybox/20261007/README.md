# Canonical multimodal provenance review

The repository contains a canonical 71,010-row non-training multimodal master used by the 5–6 October Gray-Box analyses. Its role counts exactly match the 71,010 non-training rows in the failed v3 candidate schedule. A local case-level comparison is possible for the supplementary role because the failed v3 relay delivered 2,560 rows: all 2,560 are present in the canonical 11,118-row supplementary support, with no v3-only IDs and 8,558 canonical-only IDs.

That overlap is useful, but it does not prove model-generation lineage. The local repository contains repeated SHA256 references and downstream result protocols, but no original successful creation command, executed notebook receipt, case-level canonical master transfer archive, per-row checkpoint/normalization binding, or successful VM cleanup receipt for the canonical master. The canonical results therefore remain protocol-scoped reported evidence pending lineage acceptance.

The review is a same-assistant local pass, not external scientific validation. The expired v3 relay remains `relay_incomplete` with return code 255 and `cleanup_verified=false`; it was not retried. VM process, lock and GPU state remain unknown after the 7 October IAP connection failure.
