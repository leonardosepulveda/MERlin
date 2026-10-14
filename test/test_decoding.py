import numpy as np

from merlin.util import decoding


def _labels():
    # three objects of 3, 4 and 2 pixels
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[0, :3] = 1
    labels[2, :4] = 2
    labels[4, :2] = 3
    return labels


def test_remove_labels_smaller_than_keeps_area_equal_to_minimum():
    kept = decoding.remove_labels_smaller_than(_labels(), 3)
    assert set(np.unique(kept)) == {0, 1, 2}
    assert (kept[0, :3] == 1).all()


def test_remove_labels_smaller_than_one_keeps_everything():
    labels = _labels()
    assert (decoding.remove_labels_smaller_than(labels, 1) == labels).all()


def test_extract_barcodes_keeps_area_equal_to_minimum():
    # decodedImage is -1 off barcodes; each object decodes to barcode 0
    decoded = np.full((10, 10), -1, dtype=np.int16)
    decoded[_labels() > 0] = 0
    decoded[2, 4] = -1  # keep the 4-px object separate from its row
    decoder = decoding.PixelBasedDecoder.__new__(
        decoding.PixelBasedDecoder)
    barcodes = decoder.extract_barcodes_with_index(
        decoded, np.ones((10, 10), np.float32),
        np.ones((2, 10, 10), np.float32), np.zeros((10, 10), np.float32),
        fov=0, cropWidth=0, zIndex=0, minimumArea=3)
    assert sorted(barcodes['area']) == [3, 4]
