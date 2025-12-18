# Steering vectors

This fork keeps the control vector path (aka steering vectors) enabled by default so you can read and modify the per-layer tensors directly.

## What gets initialized
- On context creation the library allocates a float32 steering tensor for each transformer layer (starting at layer 1) sized to the model's embedding dimension.
- Every tensor is zeroed; steering is effectively a no-op until you write new values.
- Initialization happens automatically when `llama_init_from_model` finishes unless the model was loaded in vocab-only mode.

## How to access the tensors
Use the C API to discover and work with the steering tensors:

```c
const int32_t start = llama_steering_layer_start(ctx);
const int32_t end   = llama_steering_layer_end(ctx);
const int32_t d     = llama_steering_n_embd(ctx);

for (int32_t il = start; il <= end; ++il) {
    struct ggml_tensor * tensor = llama_get_steering_tensor(ctx, il);
    // tensor->data points to the editable buffer; use ggml_backend_tensor_* helpers to read/write safely
}
```

Because the tensors live in the existing backend buffers, external tools (for example, a future ComfyUI node) can either map the `ggml_tensor` memory or use `ggml_backend_tensor_get/set` to exchange data without reloading the model.

## Editing notes
- All steering tensors share the model's device placement; expect GPU-resident buffers if the corresponding layer is offloaded.
- You can reset everything by applying another zero control vector via `llama_apply_adapter_cvec`.
- The API returns `nullptr` for layer 0 because control vectors start at layer 1.

These entry points are intentionally small and explicit so downstream tooling can enumerate layers, fetch the backing buffers, and push updates in place without extra file formats.
