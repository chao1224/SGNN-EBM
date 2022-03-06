1. Mapping option 1

Download STRING files first, but then need to find some manual mapping files.

```
wget https://stringdb-static.org/download/protein.info.v11.0.txt.gz
wget https://string-db.org/mapping_files/uniprot/all_organisms.uniprot_2_string.2018.tsv.gz
```

2. Mapping option 2 (Now using this)

Use API.

1. `step_01.py`: mapping from ChEMBL ID to UniProt ID
    + output files: `assay2target.tsv` (i.e., `assay2uniprot`)
2. `step_02.py`: mapping from all UniProt to STRING ID to PPI (STRING)
    + output files: `assay2target.tsv` -> `uniprot2string.tsv` -> `string_ppi_score.tsv`
3. `step_03.py`: mapping from ChEMBL assay ID to PPI
    + output files: `assay2target.tsv`, `uniprot2string.tsv`, `string_ppi_score.tsv` -> `filtered_assay_score.tsv`
4. `step_04.py`: mapping from task ID to PPI
    + output files: `filtered_assay_score.tsv` -> `filtered_task_score.tsv`
5. `step_05.py`: verify the task/assay ordering

