# This step prepares the dataset for parallel processing in a Cloud Run Job.


Loads the full review dataset.

Splits it into several equal-sized chunks based on the total number of tasks in the job.

Uses the task’s index (CLOUD_RUN_TASK_INDEX) to select the specific chunk that this task is responsible for.

Each task processes only its own chunk, which ensures:


Simple overview
full dataset
     ↓ split into N chunks
chunk_0 → task 0
chunk_1 → task 1
chunk_2 → task 2
 ...
