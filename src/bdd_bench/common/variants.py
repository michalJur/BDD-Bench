from __future__ import annotations

from pathlib import Path

VARIANTS = ("golden", "silver", "bronze")
DEFAULT_VARIANT = "bronze"
DEFAULT_DATASET_ROOT = Path("output_dataset")
VARIANT_TO_TEST_PATCH_KEY = {
    "golden": "golden_test_patch",
    "silver": "silver_test_patch",
    "bronze": "bronze_test_patch",
}
TEST_PATCH_KEY_TO_VARIANT = {value: key for key, value in VARIANT_TO_TEST_PATCH_KEY.items()}
VARIANT_TO_SUMMARY_SUBDIR = {
    "golden": "6_test_summary",
    "silver": "10_test_summary_silver",
    "bronze": "13_test_summary_bronze",
}
TEST_PATCH_KEY_TO_SUMMARY_SUBDIR = {
    VARIANT_TO_TEST_PATCH_KEY[variant]: subdir
    for variant, subdir in VARIANT_TO_SUMMARY_SUBDIR.items()
}


def normalize_variant(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VARIANTS:
        raise ValueError(f"Unsupported variant '{value}'. Expected one of: {', '.join(VARIANTS)}.")
    return normalized


def variant_for_test_patch_key(patch_key: str) -> str:
    try:
        return TEST_PATCH_KEY_TO_VARIANT[patch_key]
    except KeyError as error:
        raise ValueError(f"Unsupported test patch key: {patch_key}") from error


def test_patch_key_for_variant(variant: str) -> str:
    try:
        return VARIANT_TO_TEST_PATCH_KEY[normalize_variant(variant)]
    except KeyError as error:
        raise ValueError(f"Unsupported variant: {variant}") from error


def agent_visible_test_patch_key_for_evaluation(patch_key: str | None) -> str:
    normalized = patch_key.strip() if isinstance(patch_key, str) else ""
    if normalized == "bronze_test_patch":
        return "silver_test_patch"
    if normalized in TEST_PATCH_KEY_TO_VARIANT:
        return normalized
    return "golden_test_patch"


def summary_subdir_for_variant(variant: str) -> str:
    return VARIANT_TO_SUMMARY_SUBDIR[normalize_variant(variant)]


def summary_subdir_for_test_patch_key(patch_key: str) -> str:
    try:
        return TEST_PATCH_KEY_TO_SUMMARY_SUBDIR[patch_key]
    except KeyError as error:
        raise ValueError(f"Unsupported test patch key: {patch_key}") from error


def final_instances_filename(variant: str) -> str:
    return f"final_instances_{normalize_variant(variant)}.json"


def dataset_instances_filename() -> str:
    return "dataset_instances.json"


def failed_instances_filename(variant: str) -> str:
    return f"failed_instances_{normalize_variant(variant)}.json"


def all_fail_to_pass_tests_filename(variant: str) -> str:
    return f"all_fail_to_pass_tests_{normalize_variant(variant)}.json"


def final_instances_path(root: Path, variant: str) -> Path:
    return root / final_instances_filename(variant)


def dataset_instances_path(root: Path) -> Path:
    return root / dataset_instances_filename()


def failed_instances_path(root: Path, variant: str) -> Path:
    return root / failed_instances_filename(variant)


def all_fail_to_pass_tests_path(root: Path, variant: str) -> Path:
    return root / all_fail_to_pass_tests_filename(variant)


def default_final_instances_path(variant: str) -> Path:
    return final_instances_path(DEFAULT_DATASET_ROOT, variant)


def default_dataset_instances_path() -> Path:
    return dataset_instances_path(DEFAULT_DATASET_ROOT)


def default_failed_instances_path(variant: str) -> Path:
    return failed_instances_path(DEFAULT_DATASET_ROOT, variant)


def default_all_fail_to_pass_tests_path(variant: str) -> Path:
    return all_fail_to_pass_tests_path(DEFAULT_DATASET_ROOT, variant)
