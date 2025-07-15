import candombe
from hvo_sequence import HVO_Sequence
from pickle import load
import os
import numpy as np

#'/Users/justinbosma/Desktop/TripleStreams/candombe/candombe_annotations/UC_211.csv'
def test_create_hvo_from_annotation(dir, file):
    hvo_seq_test = candombe.create_hvo_from_annotation(dir, file)
    assert hvo_seq_test.hvo.shape == (1344, 12)
    assert hvo_seq_test.hits[3] == [0, 0, 1, 1]
    assert hvo_seq_test.velocities[3] == [0., 0., 0.032734, 0.032734]
    assert (np.allclose(hvo_seq_test.offsets[3], [0., 0., 0.12936171, 0.]) or np.allclose(hvo_seq_test.offsets[3], [0., 0., 0.12936171, 0.12936171]))
    hvo_seq_test.save('/Users/justinbosma/Desktop/TripleStreams/candombe/tests', 'hvo_test')

def test_save_correct_hvo(dir, file):
    hvo_seq_test = load(open("/Users/justinbosma/Desktop/TripleStreams/candombe/tests/hvo_test.hvo", "rb"))
    assert hvo_seq_test.hvo.shape == (1344, 12)
    assert hvo_seq_test.hits[3] == [0, 0, 1, 1]
    assert hvo_seq_test.velocities[3] == [0., 0., 0.032734, 0.032734]
    assert (np.allclose(hvo_seq_test.offsets[3], [0., 0., 0.12936171, 0.]) or np.allclose(hvo_seq_test.offsets[3], [0., 0., 0.12936171, 0.12936171]))
    os.remove("/Users/justinbosma/Desktop/TripleStreams/candombe/tests/hvo_test.hvo")

