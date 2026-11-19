"""Re-derive the LLM-decided labels and clusters with a local open model, in a separate run directory.

The three LLM steps of the labeling stages (``regrade`` math grading, ``regrade`` answer-equivalence pairs,
``extract`` short-answer extraction) are re-run with a local model through vLLM on exactly the requests sent to
Claude Haiku (the stored ``*.keys.jsonl`` / ``*.ids`` files) and with the same prompts. The results are written in
the Batch-result format, so the verdicts of both sources are read the same way.

Stages::

    python -m uqbench.labeling.local run   --models qwen3-8b [--tasks regrade pairs extract]   # GPU (vLLM)
    python -m uqbench.labeling.local agree [--models qwen3-8b]                                # agreement and kappa
    python -m uqbench.labeling.local build --models qwen3-8b --source gpt-oss-120b            # relabeled run
    python -m uqbench.labeling.local label --models gpt-oss-120b --source Qwen3-32B          # label a new run

``build`` starts from the main run, undoes the LLM decisions (LLM-graded labels listed in ``regrade.jsonl``,
extracted answers), applies the verdicts of ``--source`` (``haiku`` = the main run's own results, which rebuilds
the main labels and clusters as a check) and recomputes the answer clusters of the affected datasets. The run is
written to ``OUTPUT_DIR/relabel/<source>/<model>/``: rewritten generations (without token ids) and cluster caches;
unchanged generations and every inference signal are symlinked. Evaluate it with::

    OUTPUT_DIR=$OUTPUT_DIR/relabel/<source> RESULTS_DIR=results/boost/relabel_<source> \\
        python -m uqbench.pipeline.evaluate --model qwen3-8b

The main run is never modified.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from uqbench.config import OUTPUT_DIR, model_path, run_dir
from uqbench.io import read_jsonl, write_jsonl
from uqbench.labeling import extract, regrade
from uqbench.verify import equivalent, is_correct, normalize_qa

LOCAL_MODELS = {   # local labeler -> (path under MODELS_DIR or Hugging Face id, reasoning format)
    "gpt-oss-120b": (model_path("openai/gpt-oss-120b"), "harmony"),
    "Qwen3-32B": (model_path("Qwen/Qwen3-32B"), "think"),
}
TASKS = {   # task -> (Haiku result dir, reasoning effort, reasoning budget, answer budget, schema)
    "regrade": (regrade.ROOT, "medium", 4096, 512, regrade.SCHEMA),
    "pairs": (regrade.PAIR_ROOT, "medium", 4096, 512, regrade.PAIR_SCHEMA),
    "extract": (extract.ROOT, "low", 512, 64, extract.SCHEMA),
}
SIGNALS = ("replay", "energy", "followup", "reward", "judge", "judge_think", "judge_think_gpt-oss-120b", "prm")


def source_root(source: str) -> Path:
    return OUTPUT_DIR / "relabel" / source


def result_dirs(source: str) -> dict[str, Path]:
    """Where the verdicts of ``source`` live, per task."""
    if source == "haiku":
        return {t: v[0] for t, v in TASKS.items()}
    return {t: source_root(source) / v[0].name for t, v in TASKS.items()}


# ---------- requests (exactly those sent to Haiku) ----------
def gen_rows(model: str):
    for path in sorted((run_dir(model) / "gen").glob("*.jsonl")):
        if path.stem not in regrade.SKIP:
            yield path.stem, list(read_jsonl(path))


def model_preds(models: list[str]) -> tuple[set, dict, set]:
    """(dataset, uid, pred) of the models' responses, uid -> question, and math uid -> preds."""
    triples, questions, by_uid = set(), {}, {}
    for model in models:
        for name, rows in gen_rows(model):
            for r in rows:
                questions[(name, r["uid"])] = r
                for s in r["samples"]:
                    if s["pred"] is not None:
                        triples.add((name, r["uid"], s["pred"]))
                        if r["task"] == "math":
                            by_uid.setdefault(r["uid"], set()).add(s["pred"])
    return triples, questions, by_uid


def requests(task: str, models: list[str] | None):
    """(key, prompt) of every request Haiku received for ``task``, restricted to the responses of ``models``."""
    everything = models is None
    models = models or sorted({p.parent.parent.name for p in OUTPUT_DIR.glob("*/gen/math500.jsonl")})
    triples, questions, by_uid = model_preds(models)
    text = {uid: r["question"] for (_, uid), r in questions.items()}
    root = TASKS[task][0]
    seen = set()
    if task == "regrade":
        for path in sorted(root.glob("*.keys.jsonl")):
            for row in read_jsonl(path):
                model, name, uid, pred = row["key"]
                k = (name, uid, pred)
                if k in seen or not (everything or k in triples) or (name, uid) not in questions:
                    continue
                seen.add(k)
                r = questions[(name, uid)]
                yield list(k), regrade.PROMPT.format(question=r["question"][:4000], answer=r["answer"], pred=pred)
    elif task == "pairs":
        for path in sorted(root.glob("*.keys.jsonl")):
            for row in read_jsonl(path):
                uid, a, b = row["key"]
                k = (uid, a, b)
                if k in seen or uid not in text or not (everything or {a, b} <= by_uid.get(uid, set())):
                    continue
                seen.add(k)
                yield list(k), regrade.PAIR_PROMPT.format(question=text[uid][:4000], a=a, b=b)
    else:
        wanted = set(models)
        responses = {}
        for model in wanted:
            for name in extract.DATASETS:
                path = run_dir(model) / "gen" / f"{name}.jsonl"
                for r in read_jsonl(path) if path.exists() else []:
                    for k, s in enumerate(r["samples"]):
                        responses[f"{model}__{r['uid']}__{k}"] = (r, s["response"])
        for path in sorted(root.glob("*.ids")):
            for cid in path.read_text().split():
                if cid in seen or cid not in responses:
                    continue
                seen.add(cid)
                r, resp = responses[cid]
                yield cid, extract.PROMPT.format(question=r["question"], response=resp[-extract.TAIL:])


# ---------- local inference ----------
def engine(source: str):
    """vLLM engine, tokenizer, reasoning format and stop / answer-opening token ids of a local labeler."""
    from transformers import AutoTokenizer
    from vllm import LLM

    path, fmt = LOCAL_MODELS[source]
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=256, seed=0,
              enable_prefix_caching=True)
    if fmt == "harmony":
        stops = [tok.convert_tokens_to_ids(t) for t in ("<|end|>", "<|return|>")]
        opening = tok.encode("<|end|><|start|>assistant<|channel|>final<|message|>", add_special_tokens=False)
    else:   # think: reasoning inside <think>...</think>
        stops = [tok.convert_tokens_to_ids("</think>")]
        opening = tok.encode("\n</think>\n\n", add_special_tokens=False)
    return llm, tok, fmt, stops, opening


def answer(eng, task: str, items: list[tuple], out: Path) -> None:
    """Reason, then answer in the task's JSON schema (structured outputs); write Batch-format results to ``out``."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    llm, tok, fmt, stops, opening = eng
    _, effort, think, answer_tokens, schema = TASKS[task]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[local] {task}: {len(items)} requests", flush=True)
    if not items:
        return
    t0 = time.time()
    kw = {"reasoning_effort": effort} if fmt == "harmony" else {"enable_thinking": True}
    heads = [list(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True,
                                          return_dict=False, **kw)) for _, p in items]
    drafts = llm.generate([TokensPrompt(prompt_token_ids=h) for h in heads],
                          SamplingParams(max_tokens=think, temperature=0.0, skip_special_tokens=False,
                                         stop_token_ids=stops), use_tqdm=False)
    cont = []
    for h, d in zip(heads, drafts, strict=True):
        ids = list(d.outputs[0].token_ids)
        while ids and ids[-1] in stops:
            ids.pop()
        cont.append(TokensPrompt(prompt_token_ids=h + ids + opening))
    answers = llm.generate(cont, SamplingParams(max_tokens=answer_tokens, temperature=0.0,
                                                structured_outputs=StructuredOutputsParams(json=schema)),
                           use_tqdm=False)
    stem = f"local{len(list(out.glob('local*.results.jsonl')))}"
    rows, keys, failed = [], [], 0
    for i, ((key, _), a) in enumerate(zip(items, answers, strict=True)):
        verdict = {"custom_id": f"{stem}_{i}", "type": "succeeded"}
        try:
            verdict.update(json.loads(a.outputs[0].text))
        except json.JSONDecodeError:
            verdict["type"], failed = "errored", failed + 1
        if task == "extract":
            verdict = {"custom_id": key, "type": verdict["type"],
                       "answer": (verdict.get("answer") or "").strip() or None}
        else:
            keys.append({"custom_id": verdict["custom_id"], "key": list(key)})
        rows.append(verdict)
    write_jsonl(out / f"{stem}.results.jsonl", rows)
    if task == "extract":
        (out / f"{stem}.ids").write_text("\n".join(k for k, _ in items))
    else:
        write_jsonl(out / f"{stem}.keys.jsonl", keys)
    print(f"[local] {task}: {len(rows)} verdicts ({failed} unparsed) in {(time.time() - t0) / 60:.1f} min",
          flush=True)


def submitted(task: str, out: Path) -> set[str]:
    if task == "extract":
        return {c for p in out.glob("local*.ids") for c in p.read_text().split()}
    return {json.dumps(list(r["key"])) for p in out.glob("local*.keys.jsonl") for r in read_jsonl(p)}


def run(args) -> None:
    """Re-run Haiku's requests (``requests``) with the local model."""
    eng = engine(args.source)
    dirs = result_dirs(args.source)
    for task in args.tasks:
        done = submitted(task, dirs[task])
        answer(eng, task, [(k, p) for k, p in requests(task, args.models)
                           if (k if task == "extract" else json.dumps(list(k))) not in done], dirs[task])


def label(args) -> None:
    """Label a new run in place with a local model where ``scripts/run_labels.sh`` uses Haiku: the rules, then LLM
    grading of the math answers the rules reject, LLM answer-equivalence pairs (clusters merged in the run's cache;
    the main run's ``math_equiv/equivalent.jsonl`` is not touched) and short-answer extraction."""
    regrade.rules(argparse.Namespace(models=args.models))
    eng, dirs = engine(args.source), result_dirs(args.source)
    done = {tuple(json.loads(k))[-3:] for k in submitted("regrade", dirs["regrade"])}
    items = {}
    for key, r in regrade.pending(args.models):
        if key[1:] not in done:
            items.setdefault(key[1:], (list(key), regrade.PROMPT.format(question=r["question"][:4000],
                                                                         answer=r["answer"], pred=key[3])))
    answer(eng, "regrade", list(items.values()), dirs["regrade"])
    got = grades(args.source)
    for model in args.models:
        root, changes = run_dir(model), []
        for path, rows in regrade.math_files(model):
            for r in rows:
                for k, s in enumerate(r["samples"]):
                    if not s["correct"] and got.get((path.stem, r["uid"], s["pred"])):
                        s["correct"] = 1
                        changes.append({"dataset": path.stem, "uid": r["uid"], "sample": k, "pred": s["pred"],
                                        "answer": r["answer"], "source": f"llm:{args.source}"})
            regrade.rewrite(root, path.stem, rows, None)
        regrade.log(root, changes)
        print(f"[local] {model}: {len(changes)} responses graded correct by {args.source}", flush=True)
    done = submitted("pairs", dirs["pairs"])
    answer(eng, "pairs", [(list(k), regrade.PAIR_PROMPT.format(question=q[:4000], a=k[1], b=k[2]))
                          for k, q in regrade.pairs([args.models]) if json.dumps(list(k)) not in done], dirs["pairs"])
    table: dict[str, set] = {}
    for (uid, a, b), same in pair_verdicts(args.source).items():
        if same:
            table.setdefault(uid, set()).add(frozenset((a, b)))
    for model in args.models:
        merged_q = 0
        for path, rows in regrade.math_files(model):
            cache = run_dir(model) / "clusters" / f"{path.stem}.jsonl"
            old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)}
            new = {}
            for r in rows:
                same = table.get(r["uid"], set())
                merged = regrade.merge(old[r["uid"]], [s["pred"] for s in r["samples"]],
                                       lambda a, b, s=same: frozenset((a, b)) in s)
                if merged != old[r["uid"]]:
                    new[r["uid"]] = merged
            regrade.rewrite(run_dir(model), path.stem, rows, new)
            merged_q += len(new)
        print(f"[local] {model}: {merged_q} questions reclustered by {args.source} pairs", flush=True)
    done = submitted("extract", dirs["extract"])
    answer(eng, "extract", [(cid, extract.PROMPT.format(question=r["question"], response=resp[-extract.TAIL:]))
                            for cid, r, resp in extract.pending(args.models) if cid not in done], dirs["extract"])
    ext = extractions(args.source)
    for model in args.models:
        root = run_dir(model)
        for name in extract.DATASETS:
            path = root / "gen" / f"{name}.jsonl"
            if not path.exists():
                continue
            rows = list(read_jsonl(path))
            cache = root / "clusters" / f"{name}.jsonl"
            old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)} if cache.exists() else {}
            changed = {}
            for r in rows:
                for k, s in enumerate(r["samples"]):
                    a = ext.get(f"{model}__{r['uid']}__{k}")
                    if s["pred"] is None and a and normalize_qa(a):
                        s["pred"], s["pred_source"] = normalize_qa(a), "extracted"
                        s["correct"] = int(is_correct(s["pred"], r["answer"], r["task"]))
                        changed[r["uid"]] = r
            write_jsonl(path, rows)
            if cache.exists():
                write_jsonl(cache, ({"uid": u, "clusters": cluster([s["pred"] for s in changed[u]["samples"]],
                                                                   changed[u]["task"], set()) if u in changed else c}
                                    for u, c in old.items()))
            print(f"[local] {model}/{name}: {len(changed)} questions with extracted answers", flush=True)


# ---------- verdicts of a source ----------
def grades(source: str) -> dict[tuple, bool]:
    """(dataset, uid, answer) -> correct (a key graded for several runs counts as correct if any verdict says so)."""
    got = {}
    for key, v in regrade.results(result_dirs(source)["regrade"]).items():
        k = tuple(key[-3:])
        got[k] = got.get(k, False) or v.get("correct") is True
    return got


def pair_verdicts(source: str) -> dict[tuple, bool]:
    return {tuple(k): v.get("same") is True for k, v in regrade.results(result_dirs(source)["pairs"]).items()}


def extractions(source: str) -> dict[str, str | None]:
    out = {}
    for path in sorted(result_dirs(source)["extract"].glob("*.results.jsonl")):
        out.update({r["custom_id"]: r.get("answer") for r in read_jsonl(path)})
    return out


def kappa(a: list[bool], b: list[bool]) -> tuple[float, float]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree(args) -> None:
    other = args.source
    rows = []
    for task in args.tasks:
        if task == "regrade":
            h, o = grades("haiku"), grades(other)
        elif task == "pairs":
            h, o = pair_verdicts("haiku"), pair_verdicts(other)
        else:
            h, o = extractions("haiku"), extractions(other)
        wanted = {tuple(k) if isinstance(k, list) else k for k, _ in requests(task, args.models)}
        common = [k for k in o if k in h and k in wanted]
        if not common:
            continue
        if task == "extract":
            ids = {c: c.split("__") for c in common}
            gold = {}
            for model in {m for m, _, _ in ids.values()}:
                for name in extract.DATASETS:
                    path = run_dir(model) / "gen" / f"{name}.jsonl"
                    for r in read_jsonl(path) if path.exists() else []:
                        gold[(model, r["uid"])] = (r["answer"], r["task"])

            def grade(ans, c, ids=ids, gold=gold):
                ans = normalize_qa(ans) if ans else ""
                m, uid, _ = ids[c]
                return bool(ans) and is_correct(ans, *gold[(m, uid)])

            same_text = sum(normalize_qa(h[c] or "") == normalize_qa(o[c] or "") for c in common) / len(common)
            po, k = kappa([grade(h[c], c) for c in common], [grade(o[c], c) for c in common])
            rows.append({"task": "extract (answer text)", "n": len(common), "agreement": same_text, "kappa": None})
            rows.append({"task": "extract (graded correct)", "n": len(common), "agreement": po, "kappa": k,
                         "haiku_positive": sum(grade(h[c], c) for c in common) / len(common),
                         "local_positive": sum(grade(o[c], c) for c in common) / len(common)})
        else:
            a, b = [h[c] for c in common], [o[c] for c in common]
            po, k = kappa(a, b)
            rows.append({"task": task, "n": len(common), "agreement": po, "kappa": k,
                         "haiku_positive": sum(a) / len(a), "local_positive": sum(b) / len(b)})
    import pandas as pd

    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False, float_format="%.4f")


# ---------- relabeled run ----------
def cluster(preds: list, task: str, same: set) -> list[int]:
    """``uqbench.verify.cluster_answers`` with an explicit table of LLM-equivalent math pairs."""
    reps, labels = [], []
    for ans in preds:
        for cid, rep in enumerate(reps):
            if rep is not None and ans is not None and (frozenset((rep, ans)) in same or equivalent(rep, ans, task)):
                labels.append(cid)
                break
        else:
            reps.append(ans)
            labels.append(len(reps) - 1)
    return labels


def build(args) -> None:
    got, pairs, ext = grades(args.source), pair_verdicts(args.source), extractions(args.source)
    table: dict[str, set] = {}
    for (uid, a, b), same in pairs.items():
        if same:
            table.setdefault(uid, set()).add(frozenset((a, b)))
    root = source_root(args.source)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "math_equiv_equivalent.jsonl",
                ({"uid": u, "a": a, "b": b} for (u, a, b), s in pairs.items() if s))
    for model in args.models:
        src, dst = run_dir(model), root / model
        (dst / "gen").mkdir(parents=True, exist_ok=True)
        (dst / "clusters").mkdir(parents=True, exist_ok=True)
        for sub in SIGNALS:
            if (src / sub).exists() and not (dst / sub).exists():
                (dst / sub).symlink_to(src / sub)
        llm_flips = {(c["dataset"], c["uid"], c["sample"]) for c in read_jsonl(src / "regrade.jsonl")
                     if c["source"] == "llm"} if (src / "regrade.jsonl").exists() else set()
        log, stats = [], Counter()
        for path in sorted((src / "gen").glob("*.jsonl")):
            name = path.stem
            rows = list(read_jsonl(path))
            kind = rows[0]["task"] if rows else None
            touched = name not in regrade.SKIP and (kind == "math" or name in extract.DATASETS)
            out_gen, out_cl = dst / "gen" / path.name, dst / "clusters" / path.name
            cache = src / "clusters" / path.name
            if not touched:
                for target, origin in ((out_gen, path), (out_cl, cache)):
                    if origin.exists() and not target.exists():
                        target.symlink_to(origin)
                continue
            old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)} if cache.exists() else {}
            new = {}
            for r in rows:
                redo = False
                for k, s in enumerate(r["samples"]):
                    s["token_ids"] = []
                    if r["task"] == "math":
                        if (name, r["uid"], k) in llm_flips:   # undo the LLM grade
                            s["correct"] = 0
                        if not s["correct"] and s["pred"] is not None and got.get((name, r["uid"], s["pred"])):
                            s["correct"] = 1
                            stats["math_llm_correct"] += 1
                            log.append({"dataset": name, "uid": r["uid"], "sample": k, "pred": s["pred"],
                                        "answer": r["answer"], "source": "llm"})
                    elif r["task"] == "qa":
                        if s.get("pred_source") == "extracted":   # undo the extraction
                            s["pred"], s["correct"], redo = None, 0, True
                            s.pop("pred_source")
                        answer = ext.get(f"{model}__{r['uid']}__{k}")
                        if s["pred"] is None and answer and normalize_qa(answer):
                            s["pred"], s["pred_source"] = normalize_qa(answer), "extracted"
                            s["correct"] = int(is_correct(s["pred"], r["answer"], r["task"]))
                            redo = True
                            stats["qa_extracted"] += 1
                            stats["qa_extracted_correct"] += s["correct"]
                preds = [s["pred"] for s in r["samples"]]
                if r["task"] == "math":
                    new[r["uid"]] = cluster(preds, "math", table.get(r["uid"], set()))
                elif redo or r["uid"] not in old:
                    new[r["uid"]] = cluster(preds, r["task"], set())
            write_jsonl(out_gen, rows)
            write_jsonl(out_cl, ({"uid": u, "clusters": new.get(u, c)} for u, c in
                                 ({**old, **{u: c for u, c in new.items() if u not in old}}).items()))
            same_part = sum(partition(new.get(u, c)) == partition(c) for u, c in old.items())
            stats["questions"] += len(old)
            stats["clusters_unchanged"] += same_part
        write_jsonl(dst / "regrade.jsonl", log)
        print(f"[local] {model} ({args.source}): {dict(stats)}", flush=True)


def partition(labels: list[int]) -> frozenset:
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, []).append(i)
    return frozenset(frozenset(g) for g in groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("run", "agree", "build", "label"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", default=None)
        p.add_argument("--source", default="gpt-oss-120b")
        p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
        if name == "agree":
            p.add_argument("--out", default="")
    args = ap.parse_args()
    if args.stage in ("build", "label") and not args.models:
        ap.error(f"{args.stage} needs --models")
    {"run": run, "agree": agree, "build": build, "label": label}[args.stage](args)


if __name__ == "__main__":
    main()
