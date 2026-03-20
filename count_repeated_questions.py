import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_FILE = r"dataset_evaluacion_300_ordenado.csv"
DEFAULT_CATEGORY_COLUMN = "decision_manual"


def read_rows(csv_path: Path):
    encodings = ["utf-8-sig", "utf-8", "latin-1"]

    for encoding in encodings:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                sample = f.read(4096)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)
                rows = list(reader)
                if not rows:
                    return [], []
                return rows, reader.fieldnames or []
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("codec", b"", 0, 1, "Could not decode file with supported encodings")


def detect_question_column(fieldnames, explicit_column=None):
    if explicit_column:
        if explicit_column in fieldnames:
            return explicit_column
        lowered_map = {name.lower(): name for name in fieldnames}
        if explicit_column.lower() in lowered_map:
            return lowered_map[explicit_column.lower()]
        raise ValueError(f"Column '{explicit_column}' was not found.")

    candidate_names = [
        "question",
        "pregunta",
        "questions",
        "requirement_question",
        "user_question",
    ]

    lowered_map = {name.lower(): name for name in fieldnames}
    for candidate in candidate_names:
        if candidate in lowered_map:
            return lowered_map[candidate]

    for name in fieldnames:
        low = name.lower()
        if "question" in low or "pregunta" in low:
            return name

    raise ValueError(
        "Could not detect the question column automatically. "
        "Use --question-column COLUMN_NAME."
    )


def detect_category_column(fieldnames, explicit_column=None):
    if explicit_column:
        if explicit_column in fieldnames:
            return explicit_column
        lowered_map = {name.lower(): name for name in fieldnames}
        if explicit_column.lower() in lowered_map:
            return lowered_map[explicit_column.lower()]
        raise ValueError(f"Category column '{explicit_column}' was not found.")

    candidate_names = [
        "decision_manual",
        "category",
        "categoria",
        "ground_truth_category",
        "label",
        "class",
    ]

    lowered_map = {name.lower(): name for name in fieldnames}
    for candidate in candidate_names:
        if candidate in lowered_map:
            return lowered_map[candidate]

    for name in fieldnames:
        low = name.lower()
        if "category" in low or "categoria" in low or "categor" in low:
            return name

    raise ValueError(
        "Could not detect the category column automatically. "
        "Use --category-column COLUMN_NAME."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Count questions per category in a CSV file."
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--question-column",
        default=None,
        help="Question column name (optional).",
    )
    parser.add_argument(
        "--category-column",
        default=DEFAULT_CATEGORY_COLUMN,
        help="Category column name.",
    )
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    rows, fieldnames = read_rows(csv_path)
    if not rows:
        print("The file has no rows to analyze.")
        return

    question_column = detect_question_column(fieldnames, args.question_column)
    category_column = detect_category_column(fieldnames, args.category_column)

    questions_raw = [str(row.get(question_column, "")).strip() for row in rows]
    questions_raw = [q for q in questions_raw if q]

    if not questions_raw:
        print(f"No non-empty questions found in column '{question_column}'.")
        return

    total_questions = len(questions_raw)

    print(f"File: {csv_path}")
    print(f"Question column: {question_column}")
    print(f"Category column: {category_column}")
    print(f"Rows with non-empty questions: {total_questions}")

    category_values = []
    for row in rows:
        question_value = str(row.get(question_column, "")).strip()
        if not question_value:
            continue
        category = str(row.get(category_column, "")).strip() or "(EMPTY)"
        category_values.append(category)

    category_counts = Counter(category_values)
    print("Questions per category:")
    for category, count in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
