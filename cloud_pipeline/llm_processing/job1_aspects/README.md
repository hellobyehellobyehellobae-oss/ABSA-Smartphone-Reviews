### LLM Processing

This section explains how each Cloud Run task interacts with the LLM (Gemini 2.5 Pro) to extract structured information from raw customer reviews.

The focus here is on the conceptual flow — how prompts are built, how batches are processed, and how outputs are mapped — without going into validation or missing-row logic (which is handled separately in the llm_output_validation module).

---

### What this module does

The goal of the LLM processing stage is to transform unstructured natural-language reviews into structured JSON objects that follow a predefined schema.

Each Cloud Run task:
- Receives a chunk of the dataset (already split earlier)
- Processes that chunk in multiple batches
- Sends each batch to Gemini using a carefully designed prompt
- Receives model output for every review in the batch
- Maps local batch indices back to the global review IDs

This module is responsible only for:
- Preparing the prompt
- Sending the prompt to the LLM
- Receiving and cleaning the response
- Mapping outputs to the correct review identifiers

It does **not** perform retries or schema checks — those belong to llm_output_validation.

---

### How prompting works

Each batch of reviews is converted into a numbered text list:

0. review text...
1. review text...
2. review text...
…

The prompt template then places this text inside a structured instruction that describes the exact JSON format the model must return. The model receives one batch at a time.

The output from the LLM is a single JSON dictionary where:
- Keys represent batch-local indices  
- Values contain aspect-level sentiment analysis for each review

Example model output (conceptual):

{
  "0": { structured json for review 0 },
  "1": { structured json for review 1 },
  ...
}

---

### Mapping model output back to reviews

Because the LLM output uses batch-local indices (0, 1, 2...), these indices are mapped back to the original review IDs so the final dataset remains consistent across all Cloud Run tasks.

Mapping ensures:
- Each review keeps its correct identifier
- Downstream validation and merging work reliably
- No mixing or shuffling of reviews can occur

---

### Why this module is separate

The LLM processing module focuses only on producing clean, structured outputs from Gemini.

Other concerns — such as:
- validating the schema,
- detecting missing rows,
- retrying invalid or incomplete entries,

are intentionally handled in the `llm_output_validation` module to keep responsibilities clean and maintainable.

---

### Summary

The LLM Processing module is the core transformation step of the pipeline.  
Its purpose is to convert raw text → structured JSON using a deterministic prompt and a consistent output mapping system.

It does **not** validate, retry, or filter the results — it only produces the initial structured output for each batch.
