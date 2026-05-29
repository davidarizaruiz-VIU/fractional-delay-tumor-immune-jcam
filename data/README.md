# Patient-level data

The patient-level longitudinal PSA dataset analysed in §9 of the paper is
the Vancouver Phase II intermittent androgen-deprivation therapy (IADT)
trial cohort originally reported in:

> Bruchovsky, N., Klotz, L., Crook, J., Malone, S., Ludgate, C.,
> Morris, W. J., Gleave, M. E., Goldenberg, S. L. (2006).
> *Final results of the Canadian prospective phase II trial of
> intermittent androgen suppression for men in biochemical recurrence
> after radiotherapy for locally advanced prostate cancer.*
> Cancer 107(2), 389–395.

The dataset is redistributed in machine-readable form as the bundled
`sampleData` object of the [`tumgr`](https://cran.r-project.org/package=tumgr)
R package (Wilkerson, 2016), freely available from the Comprehensive R
Archive Network (CRAN).

## Reproducing the cohort used in the paper

1. Install R (≥ 3.5) and the `tumgr` package:

   ```r
   install.packages("tumgr")
   ```

2. Export the `sampleData` object to CSV:

   ```r
   library(tumgr)
   data(sampleData)
   write.csv(sampleData, "bruchovsky_sampleData.csv", row.names = FALSE)
   ```

3. Place the resulting `bruchovsky_sampleData.csv` file in this `data/`
   directory. The Python preprocessing pipeline
   (`src/preprocess/preprocess_bruchovsky_offphase.py`) expects this path
   and produces the cycle-1 OFF-phase sub-cohort (N = 55 patients) used
   throughout §9 of the paper.

4. The companion file `classificationsofpatients.txt`, listing the
   per-patient clinical outcome (eradication/dormancy/escape) used to
   colour-code the cohort figures, is generated automatically from
   the trial's biochemical-relapse criterion by the same preprocessing
   script.

## Data privacy

The Bruchovsky cohort is redistributed in fully de-identified form
(numeric patient IDs, treatment timestamps relative to randomisation,
PSA measurements). No personally identifying information is contained
in the dataset or in the preprocessed cohort produced by the pipeline.
The .gitignore in this repository explicitly excludes any file with
extension `.csv`, `.tsv`, or `.txt` from this `data/` directory to
prevent accidental commits of patient-level data; only this `README.md`
is tracked.
