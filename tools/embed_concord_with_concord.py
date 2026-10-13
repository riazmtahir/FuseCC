#!/usr/bin/env python3
"""Embed the ConCord test splits with CONCORD's OWN fine-tuned checkpoint.

Produces `dataset/concord/<ds>/embeddings_concord_test.npy`, row-aligned with
code_store.pkl, so it feeds straight into tools/eval_concord_mapr.py --tag concord and
gives a CONCORD MAP@R row computed by OUR evaluator (not a published or self-logged
number). This is the same-evaluator baseline row the paper wants.

Faithful to CONCORD's own pipeline (scripts/run_concord_finetune_cc_cxg.py):
- INPUT is the RAW `code` field (already SentencePiece-tokenised, space-joined) — NOT our
  detokenised text. Verified: raw tokens hit their 50,002-word vocab at 0% UNK.
- Tokenise exactly as they do: `code=' '.join(code.split()); toks=code.split()[:510];
  ids=[CLS]+convert_tokens_to_ids(toks)+[SEP]`, pad to block_size 512 with pad_id 0.
- Embedding = CLS pooling (config emb_pooler_type='cls', new_pooler=False), i.e. BertModel's
  pooler_output. We load the checkpoint's `encoder.bert.*` weights into a plain
  transformers BertModel — ConcordForEncoder's cls path returns exactly that pooler_output,
  so this is identical without depending on their custom class or its transformers version.

Alignment is asserted: code_store row i must have the same `index` as raw jsonl line i
(both are read in file order), so labels line up in the MAP@R evaluator.

Usage:
    python tools/embed_concord_with_concord.py --dataset java250
    python tools/embed_concord_with_concord.py --dataset poj104
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import argparse
import json
import os
import pickle

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

CONCORD_ROOT = _EXT_ROOT + "/CONCORD_ISSTA_23"
CONFIG_JSON = _EXT_ROOT + "/concord_assets/concord_pretrained/config.json"
RAW_ROOT = _EXT_ROOT + "/concord_assets/data"
DATA_ROOT = _REPO_ROOT + "/dataset/concord"

SPEC = {
    "java250": {
        "ckpt": "outputs/codenet_run1/checkpoint-best-map/model.bin",
        "raw": "codenet_cc/Java250_test.jsonl",
    },
    "poj104": {
        "ckpt": "outputs/poj104_reproduce_best/checkpoint-best-map/model.bin",
        "raw": "poj104_cc/test.jsonl",
    },
}
BLOCK = 512


def load_bert(ckpt_path, device):
    cfg = BertConfig.from_json_file(CONFIG_JSON)
    model = BertModel(cfg)                       # add_pooling_layer=True by default
    sd = torch.load(ckpt_path, map_location="cpu")
    # keys look like 'encoder.bert.<name>' -> strip to '<name>' for BertModel
    bert_sd = {k[len("encoder.bert."):]: v for k, v in sd.items()
               if k.startswith("encoder.bert.")}
    missing, unexpected = model.load_state_dict(bert_sd, strict=False)
    # position_ids is a non-persistent buffer in newer transformers -> allowed to differ
    real_missing = [m for m in missing if "position_ids" not in m]
    if real_missing or unexpected:
        print(f"  load_state_dict: missing={real_missing} unexpected={unexpected}")
    assert not real_missing, "checkpoint did not fully populate BertModel"
    return model.to(device).eval()


def encode(raw_codes, tok, model, device, batch=32):
    out = np.zeros((len(raw_codes), model.config.hidden_size), dtype=np.float32)
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    for s in range(0, len(raw_codes), batch):
        chunk = raw_codes[s:s + batch]
        ids_batch = []
        for code in chunk:
            toks = " ".join(code.split()).split()[:BLOCK - 2]
            ids = [cls] + tok.convert_tokens_to_ids(toks) + [sep]
            ids += [pad] * (BLOCK - len(ids))
            ids_batch.append(ids)
        input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
        attn = input_ids.ne(pad).long()          # pad id is 0, exactly their mask
        with torch.no_grad():
            pooled = model(input_ids=input_ids, attention_mask=attn).pooler_output
        out[s:s + len(chunk)] = pooled.cpu().numpy()
        if (s // batch) % 50 == 0:
            print(f"    {min(s + batch, len(raw_codes)):,}/{len(raw_codes):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SPEC))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    spec = SPEC[args.dataset]
    d = os.path.join(DATA_ROOT, args.dataset)

    with open(os.path.join(d, "code_store.pkl"), "rb") as fh:
        store = pickle.load(fh)
    store_idx = [i for i, _l, _c in store["test"]]

    # raw code in file order = code_store order; assert the indices line up
    raw_rows = [json.loads(l) for l in open(os.path.join(RAW_ROOT, spec["raw"]))]
    assert len(raw_rows) == len(store_idx), \
        f"row count mismatch {len(raw_rows)} vs {len(store_idx)}"
    raw_idx = [int(r["index"]) for r in raw_rows]
    assert raw_idx == store_idx, "raw jsonl order != code_store order — alignment broken"
    raw_codes = [r["code"] for r in raw_rows]
    print(f"{args.dataset}: {len(raw_codes):,} test samples, alignment OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BertTokenizer.from_pretrained(os.path.join(CONCORD_ROOT, "concord_tokenizer"))
    model = load_bert(os.path.join(CONCORD_ROOT, spec["ckpt"]), device)
    print(f"loaded CONCORD checkpoint {spec['ckpt']} on {device}")

    emb = encode(raw_codes, tok, model, device, args.batch_size)
    outp = os.path.join(d, "embeddings_concord_test.npy")
    np.save(outp, emb)
    print(f"-> {outp} {emb.shape}")


if __name__ == "__main__":
    main()
