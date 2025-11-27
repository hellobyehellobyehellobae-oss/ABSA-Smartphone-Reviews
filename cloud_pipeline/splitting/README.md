This step prepares the dataset for parallel processing in a Cloud Run Job.

What it does

Loads the full review dataset.

Splits it into several equal-sized chunks based on the total number of tasks in the job.

Uses the task’s index (CLOUD_RUN_TASK_INDEX) to select the specific chunk that this task is responsible for.

Why it matters

Each task processes only its own chunk, which ensures:

No duplicates

No missing reviews

Even distribution of work across parallel tasks

Simple overview
full dataset
     ↓ split into N chunks
chunk_0 → task 0
chunk_1 → task 1
chunk_2 → task 2
 ...


This module’s only purpose is to give each Cloud Run task its own portion of the dataset so the entire pipeline can run in parallel efficiently.
