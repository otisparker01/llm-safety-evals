"""Fast offline checks for the grader-gaming serial-probe mechanism control (no
model / API calls)."""

from training.grader_gaming.config import TOPICS
from training.grader_gaming import serial_probe


def test_serial_records_form_a_matched_2x2():
    recs = serial_probe.serial_records("m", n_per_cell=2)
    conditions = {r["condition"] for r in recs}
    assert conditions == {"direct_low", "serial_low", "direct_high", "serial_high"}
    # every cell present for every topic, split real/fake
    for topic in TOPICS:
        for cond in conditions:
            cell = [r for r in recs if r["topic"] == topic and r["condition"] == cond]
            assert len(cell) == 4, (topic, cond)          # 2 per-cell x (real, fake)
    # scrutiny tag matches the condition; mode is direct/serial
    assert all(r["scrutiny"] == ("low" if r["condition"].endswith("low") else "high") for r in recs)
    assert all(r["mode"] in ("direct", "serial") for r in recs)


def test_serial_high_and_low_differ_only_by_speciality():
    # For a fixed topic, serial_high (matching speciality) and serial_low (a clearly
    # non-matching one) must share identical framing wording except the speciality — so
    # the only thing that varies is whether the field matches, i.e. the inference.
    for topic in TOPICS:
        hi = serial_probe._serial_framing(serial_probe.SPECIALITY[topic])
        lo = serial_probe._serial_framing(
            serial_probe.SPECIALITY[serial_probe.NON_MATCH_TOPIC[topic]])
        assert hi != lo
        assert serial_probe.NON_MATCH_TOPIC[topic] != topic         # genuinely different field
        # identical up to the trailing "Their speciality is X." clause
        assert hi.split("Their speciality is")[0] == lo.split("Their speciality is")[0]


def test_analyse_reports_a_serial_lift():
    # Synthetic: only serial_high fires grader-reasoning; analyse must surface it as a
    # positive high-scrutiny serial lift over the matched direct_high cell.
    records = []
    for cond in ("direct_low", "serial_low", "direct_high", "serial_high"):
        fires = cond == "serial_high"
        for j in range(10):
            records.append({
                "model": "m", "condition": cond, "mode": cond.split("_")[0],
                "scrutiny": serial_probe._SCRUTINY[cond], "is_real": j % 2 == 1,
                "completion": ("<think>this will be checked and verified</think> ok."
                               if fires else "<think>ordinary</think> ok."),
            })
    report = serial_probe.analyse(records)
    hi_serial = report[("m", "serial_high")]["grader_reasoning_rate"]
    hi_direct = report[("m", "direct_high")]["grader_reasoning_rate"]
    assert hi_serial > hi_direct
