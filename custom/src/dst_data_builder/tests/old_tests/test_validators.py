import pytest
from dst_data_builder.validators.structure_validator import StructureValidator
from dst_data_builder.validators.timestamps_validator import TimestampsValidator
from dst_data_builder.validators.id_validator import IdValidator


def test_structure_validator_simple_pass():
    valid_dst = {
        "steps": [
            {"step_id": "S1", "name": "Step 1", "substeps": []}
        ]
    }

    v = StructureValidator()
    ok, msg = v.validate(valid_dst)
    assert ok


def test_timestamps_validator_pass_and_fail():
    valid_dst = {"steps": [{"step_id": "S1", "name": "Step 1", "timestamps": {"start_ts": 0, "end_ts": 1}}]}
    invalid_dst = {"steps": [{"step_id": "S1", "name": "Step 1", "timestamps": {"start_ts": 5, "end_ts": 1}}]}

    vt = TimestampsValidator(enable_post_processing=False)
    ok, msg = vt.validate(valid_dst)
    assert ok

    ok, msg = vt.validate(invalid_dst)
    assert not ok


def test_id_validator_basic_pass():
    dst = {
        "steps": [
            {
                "step_id": "S1",
                "name": "Step 1",
                "timestamps": {"start_ts": 0, "end_ts": 10},
                "substeps": [
                    {
                        "sub_id": "S1.1",
                        "name": "Sub 1",
                        "timestamps": {"start_ts": 1, "end_ts": 9},
                        "actions": [
                            {"act_id": "S1.1.a", "name": "Act A", "timestamps": {"start_ts": 1, "end_ts": 2}},
                            {"act_id": "S1.1.b", "name": "Act B", "timestamps": {"start_ts": 3, "end_ts": 4}},
                        ],
                    }
                ],
            }
        ]
    }

    v = IdValidator()
    ok, msg = v.validate(dst)
    assert ok, msg


def test_id_validator_duplicate_fail():
    dst = {
        "steps": [
            {
                "step_id": "S1",
                "name": "Step 1",
                "timestamps": {"start_ts": 0, "end_ts": 10},
                "substeps": [
                    {
                        "sub_id": "S1.1",
                        "name": "Sub 1",
                        "timestamps": {"start_ts": 1, "end_ts": 9},
                        "actions": [
                            {"act_id": "S1.1.a", "name": "Act A", "timestamps": {"start_ts": 1, "end_ts": 2}},
                            {"act_id": "S1.1.a", "name": "Act A dup", "timestamps": {"start_ts": 3, "end_ts": 4}},
                        ],
                    }
                ],
            }
        ]
    }

    v = IdValidator()
    ok, msg = v.validate(dst)
    assert not ok
    assert "duplicate" in msg.lower()


def test_id_validator_bad_format_and_parent_mismatch():
    # bad act id format
    dst1 = {
        "steps": [
            {"step_id": "S1", "name": "Step 1", "timestamps": {"start_ts": 0, "end_ts": 5}, "substeps": [
                {"sub_id": "S1.1", "name": "Sub 1", "timestamps": {"start_ts": 1, "end_ts": 4}, "actions": [
                    {"act_id": "BAD_ID", "name": "Act", "timestamps": {"start_ts": 1, "end_ts": 2}}
                ]}
            ]}
        ]
    }
    v = IdValidator()
    ok, msg = v.validate(dst1)
    assert not ok

    # parent mismatch: action claims different step/sub
    dst2 = {
        "steps": [
            {"step_id": "S1", "name": "Step 1", "timestamps": {"start_ts": 0, "end_ts": 10}, "substeps": [
                {"sub_id": "S1.1", "name": "Sub 1", "timestamps": {"start_ts": 1, "end_ts": 4}, "actions": [
                    {"act_id": "S2.1.a", "name": "Act", "timestamps": {"start_ts": 1, "end_ts": 2}}
                ]}
            ]}
        ]
    }
    ok, msg = v.validate(dst2)
    assert not ok


def test_id_validator_ordering_mismatch():
    # action ids not increasing with start_ts
    dst = {
        "steps": [
            {
                "step_id": "S1",
                "name": "Step 1",
                "timestamps": {"start_ts": 0, "end_ts": 10},
                "substeps": [
                    {
                        "sub_id": "S1.1",
                        "name": "Sub 1",
                        "timestamps": {"start_ts": 1, "end_ts": 9},
                        "actions": [
                            {"act_id": "S1.1.b", "name": "Act B", "timestamps": {"start_ts": 1, "end_ts": 2}},
                            {"act_id": "S1.1.a", "name": "Act A", "timestamps": {"start_ts": 5, "end_ts": 6}},
                        ],
                    }
                ],
            }
        ]
    }

    v = IdValidator()
    ok, msg = v.validate(dst)
    assert not ok
    assert "ID ordering mismatch" in msg

