# Transformers (BERT / BART) TF-TRT example

Benchmark Hugging Face encoder / encoder-decoder models exported to SavedModel,
then accelerated with TF-TRT.

> Full interactive guide: [`TENSORRT_DOC.md`](../../../TENSORRT_DOC.md)

## Export models from Hugging Face

```bash
# Writes under /models/<name>/pb_model by default
python generate_save_models_from_hf.py
```

Treat Hub downloads as untrusted until verified. After export:

```bash
python ../hash_saved_model.py /models/bert_base_uncased/pb_model
```

## Ready-to-use scripts

```bash
./scripts/bert_base_uncased.sh \
  --input_saved_model_dir=/models \
  --use_tftrt --precision=FP16
```

See `scripts/` for `bert_*` and `bart_*` wrappers. Flags are forwarded safely
(no `eval`) via `scripts/base_script.sh`.
