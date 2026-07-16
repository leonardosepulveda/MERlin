def test_get_analysis_tasks(simple_data, simple_task):
    assert len(simple_data.get_analysis_tasks()) == 0
    simple_task.save()
    assert len(simple_data.get_analysis_tasks()) == 1
    assert simple_data.get_analysis_tasks()[0]\
           == simple_task.get_analysis_name()


def test_dataset_get_z_positions_default_unchanged(simple_merfish_data):
    assert simple_merfish_data.get_z_positions() == [0]
    assert simple_merfish_data.z_index_to_position(0) == 0
    assert simple_merfish_data.position_to_z_index(0) == 0


def test_dataset_ragged_get_z_positions_per_fov(ragged_merfish_data):
    assert ragged_merfish_data.get_z_positions(0) == [0, 1, 2, 3]
    assert ragged_merfish_data.get_z_positions(1) == [0, 1]
    assert ragged_merfish_data.get_z_positions(2) == [0, 1, 2]
    assert ragged_merfish_data.get_z_positions(3) == [0, 1, 2, 3]


def test_dataset_ragged_z_index_to_position_per_fov(ragged_merfish_data):
    # fov 1's available z range is [0, 1] -- index 1 resolves to z position
    # 1 for this fov even though the dataset-wide range is [0, 1, 2, 3]
    assert ragged_merfish_data.z_index_to_position(1, fov=1) == 1
    assert ragged_merfish_data.position_to_z_index(1, fov=1) == 1
    # the dataset-wide (fov=None) range is unaffected
    assert ragged_merfish_data.z_index_to_position(3) == 3
