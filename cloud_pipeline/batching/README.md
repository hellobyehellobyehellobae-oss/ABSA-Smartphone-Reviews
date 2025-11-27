# Batching Module

### This module splits the task’s dataset into smaller batches and processes them asynchronously.


Breaks the task’s data into batches (for example, 30–50 reviews per batch).

Controls how many batches run at the same time (using a concurrency limit).

Sends each batch to the processing function (LLM call) one by one or in parallel.

Returns the results from all batches.

### Why this is needed

Processing the entire dataset at once is too slow and too large for the model.
Batching makes the pipeline:

faster

more stable

more efficient

easier to scale
