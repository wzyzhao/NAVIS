import pytest

from common.macro_metrics import seven_sequence_mean
from experiments.ablations.aggregate_macro import METRICS, SEQUENCES, VARIANTS, aggregate


def summaries():
    return [
        {
            "sequence": sequence,
            "results": [
                {"variant": variant, **dict.fromkeys(METRICS, index / 10), "expected_roi_groups": 10**index}
                for variant in VARIANTS
            ],
        }
        for index, sequence in enumerate(SEQUENCES, start=1)
    ]


def test_equal_sequence_weights_despite_unequal_roi_counts():
    output = aggregate(summaries())
    assert output["macro"][0]["current_roi_ratio"] == pytest.approx(0.4)


def test_missing_or_duplicate_sequence_is_rejected():
    inputs = summaries()
    with pytest.raises(ValueError):
        aggregate(inputs[:-1])
    inputs[-1] = inputs[0]
    with pytest.raises(ValueError):
        aggregate(inputs)


def test_missing_measurement_does_not_become_six_sequence_average():
    inputs = summaries()
    inputs[0]["results"][0]["avg_selected_quality_rank"] = None
    assert aggregate(inputs)["macro"][0]["avg_selected_quality_rank"] is None


def test_missing_variant_is_rejected():
    inputs = summaries()
    inputs[0]["results"] = inputs[0]["results"][1:]
    with pytest.raises(ValueError):
        aggregate(inputs)


def test_codec_macro_requires_all_seven_valid_sequence_metrics():
    rows = [{"sequence": sequence, "psnr": index} for index, sequence in enumerate(SEQUENCES)]
    assert seven_sequence_mean(rows, "psnr") == 3
    assert seven_sequence_mean(rows[:-1], "psnr") is None
    rows[-1]["psnr"] = None
    assert seven_sequence_mean(rows, "psnr") is None
