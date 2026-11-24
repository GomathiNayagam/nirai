import os, numpy as np, pytest
from nirai.gate_cnn import load_cnn_gate
MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "nirai_gate_cnn_1.0s.pt")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="model file not present")
def _windows(n, seed=0, sd=8.0):
    r = np.random.default_rng(seed); return (r.normal(0, sd, (n, 5000)).cumsum(1) * 0.05 + 100 + r.normal(0, sd, (n, 5000))).astype(np.float32)
def test_scores_are_probabilities_and_deterministic():
    g = load_cnn_gate(MODEL); w = _windows(600); ref = g.run_reference(w); p = g.score(w, ref, "native"); q = g.score(w, ref, "native")
    assert p.shape == (600,) and np.all((p >= 0) & (p <= 1)) and np.allclose(p, q)
def test_window_and_reference():
    g = load_cnn_gate(MODEL); assert g.window == 5000; w = _windows(520); assert g.run_reference(w).shape == (4,)
    assert np.allclose(g.score(np.hstack([w, w[:, :100]]), g.run_reference(w)), g.score(w, g.run_reference(w)))      # extra samples beyond the window are ignored
def test_scores_depend_on_the_signal():
    g = load_cnn_gate(MODEL); good = _windows(300, 1, 2.0); bad = _windows(300, 2, 30.0); ref = g.run_reference(good); assert g.score(good, ref).mean() != g.score(bad, ref).mean()
