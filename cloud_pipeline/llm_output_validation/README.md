### 1. Schema Validation (per-review validation)
Goal:
Guarantee that every returned object has:

overall_sentiment

aspects (a list)

each aspect with:
sentiment and summary

\
If a review does not match the schema:

that review is marked as invalid - its entry is removed from the batch output - it is added to the list of rows to retry

This ensures the pipeline never accepts malformed JSON.

### 2. Missing-Row Detection

Even if the model returns valid JSON, it may skip some reviews.
The pipeline checks this by comparing:

all expected review_ids in the batch

vs. review_ids returned by the model

If a review_id is not found in the output, it becomes a missing row.

This catches cases where the model:

outputs fewer items than requested

outputs empty dictionaries

silently ignores a review

✔ Missing rows are also added to the retry list.

### 3. Retry Logic (max 3 retries)

After gathering all invalid + missing rows, a mini-batch is created containing only those review_ids.

This mini-batch is sent back to the model again.

After each retry:

the newly returned rows are merged into the existing batch output

the pipeline re-checks the batch for both:

schema invalid rows

missing rows

Retries continue until:

all rows are valid
or

retry limit is reached

Max retries: 3 attempts

If after 3 retries some rows still:

fail schema validation
or

are not returned by the model

then:

The batch is marked as “partial fail”
Those specific rows remain missing in the final output
The batch still continues and the rest of the valid data is stored.

### 4. Final Outcome

At the end of the process, each batch produces:

a dictionary of valid review_id → extracted data

(optionally) some missing rows if they failed all retries

The batch output is appended to all_results, and later merged across tasks.

✔ The system ensures maximum completeness without stalling the entire job.
✔ Only specific failed rows are lost, not whole batches.
✔ This keeps the pipeline robust in real-world LLM usage.
