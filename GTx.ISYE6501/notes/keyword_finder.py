# keyword_finder.py
# Canonical-concept deck builder for OMSA ISyE6501.
# - Uses a curated glossary (keywords + formal + plain definitions) per module.
# - Optionally attaches a few professor-support sentences from transcript .txt files.
# Outputs per-module Markdown + Anki TSV, plus a combined Anki TSV.

import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json

import pandas as pd
from tqdm import tqdm


# ----------------------------
# 1) Curated glossary (edit this)
# ----------------------------
# Structure:
# GLOSSARY[module_number] = [
#   {"term": "...", "formal": "...", "plain": "...", "aliases": ["...", "..."]},
# ]
GLOSSARY: Dict[int, List[Dict]] = {
    1: [
        {
            "term": "Analytics",
            "formal": "Analytics is the use of data, models, and computation to support better decisions and understanding.",
            "plain": "Analytics is using data to answer questions and make better choices.",
            "aliases": ["data analytics"],
        },
        {
            "term": "Descriptive analytics",
            "formal": "Descriptive analytics summarizes what happened using reporting, aggregation, and visualization.",
            "plain": "Descriptive analytics tells you what happened.",
            "aliases": ["descriptive"],
        },
        {
            "term": "Predictive analytics",
            "formal": "Predictive analytics uses historical data to estimate or forecast unknown outcomes.",
            "plain": "Predictive analytics predicts what will happen.",
            "aliases": ["predictive"],
        },
        {
            "term": "Prescriptive analytics",
            "formal": "Prescriptive analytics recommends actions by optimizing objectives under constraints.",
            "plain": "Prescriptive analytics tells you what to do to get the best result.",
            "aliases": ["prescriptive", "optimization"],
        },
        {
            "term": "Supervised learning",
            "formal": "Supervised learning learns a mapping from inputs (features) to known outputs (labels) using labeled examples.",
            "plain": "You learn from examples where the right answer is already known.",
            "aliases": ["supervised"],
        },
        {
            "term": "Unsupervised learning",
            "formal": "Unsupervised learning finds structure in data without labeled outcomes (e.g., clustering).",
            "plain": "You find patterns without being told the right answers.",
            "aliases": ["unsupervised", "clustering"],
        },
    ],
    2: [
        {
            "term": "Classification",
            "formal": "Classification is the task of assigning an input to one of a fixed set of categories using its features.",
            "plain": "Classification is sorting things into categories based on their measurements.",
            "aliases": ["classify", "classification problem"],
        },
        {
            "term": "Classifier",
            "formal": "A classifier is a model or rule that maps feature vectors to class labels.",
            "plain": "A classifier is the rule the computer uses to pick a category.",
            "aliases": ["classification model"],
        },
        {
            "term": "Binary classification",
            "formal": "Binary classification predicts one of two possible class labels.",
            "plain": "Binary classification is a yes/no (two-choice) classification.",
            "aliases": ["two-class", "two classes"],
        },
        {
            "term": "Multiclass classification",
            "formal": "Multiclass classification predicts one of three or more class labels.",
            "plain": "Multiclass classification chooses among 3+ categories.",
            "aliases": ["multi-class", "multiple classes"],
        },
        {
            "term": "Features",
            "formal": "Features are measurable attributes used as inputs to a model.",
            "plain": "Features are the pieces of information you feed into the model.",
            "aliases": ["predictors", "inputs", "attributes"],
        },
        {
            "term": "Label (class label)",
            "formal": "A label is the target category assigned to an example (the output to predict).",
            "plain": "The label is the correct category name.",
            "aliases": ["class", "target", "output"],
        },
        {
            "term": "Training set",
            "formal": "The training set is the labeled data used to fit a model.",
            "plain": "Training data is what the model learns from.",
            "aliases": ["training data"],
        },
        {
            "term": "Test set",
            "formal": "The test set is held-out data used only to evaluate a final model.",
            "plain": "Test data checks how well the model works on new data.",
            "aliases": ["test data"],
        },
        {
            "term": "Validation set",
            "formal": "A validation set is held-out data used during model selection/tuning.",
            "plain": "Validation data helps you choose settings or compare models.",
            "aliases": ["validation data"],
        },
        {
            "term": "Decision boundary",
            "formal": "A decision boundary separates regions of feature space assigned to different classes.",
            "plain": "It’s the line/surface that divides one category from another.",
            "aliases": ["separating line", "boundary"],
        },
        {
            "term": "Misclassification",
            "formal": "Misclassification occurs when the predicted label differs from the true label.",
            "plain": "A misclassification is a wrong category guess.",
            "aliases": ["classification error", "mistake"],
        },
        {
            "term": "Cost-sensitive classification",
            "formal": "Cost-sensitive classification accounts for unequal costs of different error types.",
            "plain": "Some mistakes are worse than others, so you treat them differently.",
            "aliases": ["cost sensitive", "unequal costs"],
        },
        {
            "term": "k-Nearest Neighbors (KNN)",
            "formal": "KNN classifies a point by taking the majority label among its k closest training points under a distance metric.",
            "plain": "Look at the closest examples and vote on the category.",
            "aliases": ["knn", "k nearest neighbors", "nearest neighbor"],
        },
        {
            "term": "Distance metric",
            "formal": "A distance metric defines how ‘close’ two points are in feature space (e.g., Euclidean distance).",
            "plain": "A distance metric is how you measure closeness between two examples.",
            "aliases": ["euclidean distance", "distance"],
        },
        {
            "term": "Feature scaling",
            "formal": "Feature scaling rescales features so magnitudes are comparable, often improving distance-based and margin-based methods.",
            "plain": "Scaling puts inputs on similar ranges so one big-number feature doesn’t dominate.",
            "aliases": ["scaling", "rescaling"],
        },
        {
            "term": "Standardization",
            "formal": "Standardization rescales a feature to have mean 0 and standard deviation 1 (z-score).",
            "plain": "Standardization turns values into ‘how many standard deviations from average’.",
            "aliases": ["z-score", "z score"],
        },
        {
            "term": "Support Vector Machine (SVM)",
            "formal": "An SVM finds a separating hyperplane that maximizes the margin between classes; soft-margin allows some errors.",
            "plain": "SVM draws the best separating line by keeping the widest gap between groups.",
            "aliases": ["svm", "support vector machine"],
        },
        {
            "term": "Hyperplane",
            "formal": "A hyperplane is a linear decision boundary in a possibly high-dimensional space.",
            "plain": "A hyperplane is a ‘line’ that separates groups, even in many dimensions.",
            "aliases": ["separating hyperplane"],
        },
        {
            "term": "Margin",
            "formal": "Margin is the distance from the decision boundary to the nearest training points; SVM maximizes it.",
            "plain": "Margin is the safety gap between the boundary and the closest points.",
            "aliases": ["maximum margin"],
        },
        {
            "term": "Kernel (kernel trick)",
            "formal": "A kernel implicitly maps data into a higher-dimensional space so a linear separator can represent nonlinear boundaries.",
            "plain": "A kernel is a math trick that lets a straight-line method draw curved boundaries.",
            "aliases": ["kernel trick", "rbf kernel", "polynomial kernel", "kernel"],
        },
    ],
    3: [
        {
            "term": "Model validation",
            "formal": "Model validation estimates how well a trained model will generalize to new data.",
            "plain": "Validation checks if your model will work on unseen data.",
            "aliases": ["validation"],
        },
        {
            "term": "Overfitting",
            "formal": "Overfitting occurs when a model fits noise in the training data and performs poorly on new data.",
            "plain": "Overfitting is memorizing the training set instead of learning the real pattern.",
            "aliases": ["over fit", "over-fit"],
        },
        {
            "term": "Underfitting",
            "formal": "Underfitting occurs when a model is too simple to capture the true structure in the data.",
            "plain": "Underfitting is being too simple to learn the pattern.",
            "aliases": ["under fit", "under-fit"],
        },
        {
            "term": "Train/validation/test split",
            "formal": "A data split partitions data into training, validation, and test sets for fitting, tuning, and final evaluation.",
            "plain": "You learn on one part, tune on another, and score on a final untouched part.",
            "aliases": ["data split", "holdout", "hold-out"],
        },
        {
            "term": "Cross-validation",
            "formal": "Cross-validation repeatedly splits data into train/validation folds to reduce evaluation variance and tune models more reliably.",
            "plain": "Cross-validation tests the model multiple times on different splits to get a more stable score.",
            "aliases": ["cross validation", "cv", "k-fold", "k fold"],
        },
        {
            "term": "Bias-variance tradeoff",
            "formal": "Bias-variance tradeoff describes how model complexity affects systematic error (bias) and sensitivity to data noise (variance).",
            "plain": "Simple models miss patterns; complex models chase noise. You balance the two.",
            "aliases": ["bias variance", "bias/variance"],
        },
    ],
    4: [
        {
            "term": "Clustering",
            "formal": "Clustering groups data points into clusters so points in the same cluster are more similar than points in different clusters.",
            "plain": "Clustering groups similar things together without labels.",
            "aliases": ["cluster", "clustering problem"],
        },
        {
            "term": "k-Means",
            "formal": "k-means partitions data into k clusters by minimizing within-cluster sum of squared distances to cluster centers.",
            "plain": "k-means picks k centers and assigns points to the closest center, repeating until it settles.",
            "aliases": ["k means", "kmeans"],
        },
        {
            "term": "Cluster centroid",
            "formal": "A centroid is the mean of points assigned to a cluster (the cluster center in k-means).",
            "plain": "The centroid is the ‘average point’ of a cluster.",
            "aliases": ["centroid", "cluster center", "center"],
        },
        {
            "term": "Within-cluster sum of squares (WCSS)",
            "formal": "WCSS measures cluster tightness by summing squared distances of points to their assigned centroids.",
            "plain": "WCSS measures how spread out points are inside clusters.",
            "aliases": ["wcss", "sum of squares", "within cluster"],
        },
    ],
}


# ----------------------------
# 2) Transcript parsing + support extraction
# ----------------------------
MODULE_RE = re.compile(r"_M(?P<module>\d+)L(?P<lecture>\d+)_", re.IGNORECASE)

INTRO_CUES = re.compile(
    r"\b(in this lesson|in this lecture|we'?ll talk about|we are going to talk about|today we will|in this video)\b",
    re.IGNORECASE,
)

JUNK_CUES = re.compile(
    r"\b(google stock|stock price|traffic lights|oil tanker|restaurant|coupon|political|banner ad|airline|snow storm)\b",
    re.IGNORECASE,
)

def get_module_lecture(filename: str) -> Tuple[int, int]:
    m = MODULE_RE.search(filename)
    if not m:
        return -1, -1
    return int(m.group("module")), int(m.group("lecture"))

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

def clean_sentence(s: str) -> str:
    s = re.sub(r"^(so|okay|all right|now|well|and|but|then)\b[:, ]*", "", s.strip(), flags=re.IGNORECASE)
    return s.strip()

def term_patterns(term: str, aliases: List[str]) -> List[re.Pattern]:
    pats = [term] + (aliases or [])
    out = []
    for p in pats:
        p = p.strip()
        if not p:
            continue
        out.append(re.compile(rf"\b{re.escape(p.lower())}\b", re.IGNORECASE))
    return out

def pick_support_sentences(
    term: str,
    aliases: List[str],
    lecture_sentences: List[Tuple[int, str]],
    max_support: int = 3,
) -> List[str]:
    pats = term_patterns(term, aliases)
    hits = []
    for lec, s in lecture_sentences:
        s2 = clean_sentence(s)
        low = s2.lower()
        if JUNK_CUES.search(low):
            continue
        if INTRO_CUES.search(low):
            continue
        if len(s2.split()) < 8:
            continue
        if len(s2.split()) > 45:
            continue
        if any(p.search(low) for p in pats):
            hits.append(s2)

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for h in hits:
        k = h.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(h)
        if len(dedup) >= max_support:
            break
    return dedup


# ----------------------------
# 3) Outputs
# ----------------------------
@dataclass
class Card:
    module: int
    term: str
    formal: str
    plain: str
    support: List[str]
    lectures_seen: List[int]

def write_module_outputs(out_dir: Path, module: int, cards: List[Card]) -> None:
    # Markdown
    md = [f"# Module {module} — Key concepts\n\n"]
    for c in cards:
        md.append(f"## {c.term}\n\n")
        md.append(f"- Definition (plain): {c.plain}\n")
        md.append(f"- Definition (formal): {c.formal}\n")
        if c.support:
            md.append(f"- Support (professor phrasing):\n")
            for s in c.support:
                md.append(f"  - {s}\n")
        md.append("\n")
    (out_dir / f"Module_{module}.md").write_text("".join(md), encoding="utf-8")

    # Anki TSV
    rows = []
    for c in cards:
        front = f"Module {module} — {c.term}"
        back = f"Definition (plain): {c.plain}\n\nDefinition (formal): {c.formal}"
        if c.support:
            back += "\n\nSupport:\n" + "\n".join(f"- {s}" for s in c.support)
        rows.append({"Front": front, "Back": back})
    pd.DataFrame(rows).to_csv(out_dir / f"Module_{module}_anki.tsv", sep="\t", index=False)

    # JSON (debug)
    with open(out_dir / f"Module_{module}.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cards], f, indent=2, ensure_ascii=False)

def write_all_deck(out_dir: Path, module_cards: Dict[int, List[Card]]) -> None:
    rows = []
    for module in sorted(module_cards.keys()):
        for c in module_cards[module]:
            front = f"Module {module} — {c.term}"
            back = f"Definition (plain): {c.plain}\n\nDefinition (formal): {c.formal}"
            if c.support:
                back += "\n\nSupport:\n" + "\n".join(f"- {s}" for s in c.support)
            rows.append({"Front": front, "Back": back})
    pd.DataFrame(rows).to_csv(out_dir / "ALL_modules_anki.tsv", sep="\t", index=False)

def write_coverage_report(out_dir: Path, module_cards: Dict[int, List[Card]]) -> None:
    # For each module, list terms with 0 support hits
    lines = ["# Coverage report\n\n"]
    for module in sorted(module_cards.keys()):
        missing = [c.term for c in module_cards[module] if not c.support]
        lines.append(f"## Module {module}\n\n")
        lines.append(f"- Total terms: {len(module_cards[module])}\n")
        lines.append(f"- Terms with no transcript support hits: {len(missing)}\n\n")
        for t in missing[:50]:
            lines.append(f"  - {t}\n")
        lines.append("\n")
    (out_dir / "COVERAGE.md").write_text("".join(lines), encoding="utf-8")


# ----------------------------
# 4) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Folder with OMSA_... .txt transcript files")
    ap.add_argument("--out-dir", required=True, help="Output folder")
    ap.add_argument("--max-support", type=int, default=3, help="Support sentences per term")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect sentences by module
    module_sentences: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    txts = sorted(in_dir.glob("*.txt"))

    for p in tqdm(txts, desc="Reading transcripts"):
        module, lecture = get_module_lecture(p.name)
        if module < 0:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw)
        for s in split_sentences(text):
            s = clean_sentence(s)
            if len(s.split()) < 6:
                continue
            module_sentences[module].append((lecture, s))

    # Build cards per module from glossary
    module_cards: Dict[int, List[Card]] = {}

    for module in sorted(GLOSSARY.keys()):
        items = GLOSSARY[module]
        cards: List[Card] = []
        lecture_sents = module_sentences.get(module, [])
        lectures_seen = sorted(set(lec for lec, _ in lecture_sents))

        for it in items:
            term = it["term"]
            formal = it["formal"].strip()
            plain = it["plain"].strip()
            aliases = it.get("aliases", [])

            support = pick_support_sentences(term, aliases, lecture_sents, max_support=args.max_support)

            cards.append(
                Card(
                    module=module,
                    term=term,
                    formal=formal,
                    plain=plain,
                    support=support,
                    lectures_seen=lectures_seen,
                )
            )

        module_cards[module] = cards
        write_module_outputs(out_dir, module, cards)

    write_all_deck(out_dir, module_cards)
    write_coverage_report(out_dir, module_cards)


if __name__ == "__main__":
    main()
