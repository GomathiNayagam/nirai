"""1D-CNN pass/fail gate (34k parameters) that reads the first `window` samples (default 5000 = 1.0 s at 5 kHz) of a read and returns P(pass), i.e. 1 - P(mean Q < 10).
Inputs per read: the window of pA values (median/MAD normalised, clipped to +-6) and 10 scalars: 3 run-relative noise features + run-relative median shift + 3 log noise features
(all standardised with the training mean/std) + a 3-way kit one-hot (0 native barcode, 1 rapid barcode, 2 ligation).  'Run-relative' means divided by / minus the median of the same
features over the FIRST 500 reads of the run (arrival order, reads with a full window), so the gate needs those 500 reads before it can score (hold them, as CalibratedGate holds its warm-up).
Use with CalibratedGate for the threshold:   g = load_cnn_gate(path); ref = g.run_reference(first_500_windows); p_pass = g.score(windows, ref, kit)."""
import numpy as np, torch, torch.nn as nn
from .features import signal_features
KITS = {"native": 0, "rapid": 1, "ligation": 2}
def normalize(raw):
    raw = np.asarray(raw, np.float32); med = np.median(raw, 1, keepdims=True); mad = np.median(np.abs(raw - med), 1, keepdims=True) + 1e-3
    return np.clip((raw - med) / (1.4826 * mad), -6, 6).astype(np.float16)
def scalars(raw):
    raw = np.asarray(raw, np.float32); return np.array([signal_features(r) + [float(np.median(r))] for r in raw], np.float32)
def relative(S, ref):
    S2 = np.zeros_like(S); S2[:, :3] = S[:, :3] / (ref[:3] + 1e-9); S2[:, 3] = (S[:, 3] - ref[3]) / 10
    return np.hstack([S2, np.log(S[:, :3] + 1e-6)])
class Net(nn.Module):
    def __init__(s, ns=10):
        super().__init__(); c = lambda i, o, k, st: nn.Sequential(nn.Conv1d(i, o, k, st, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        s.conv = nn.Sequential(c(1, 16, 9, 2), c(16, 32, 9, 4), c(32, 48, 9, 4), c(48, 64, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(64 + ns, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x, sc): return s.head(torch.cat([s.conv(x), sc], 1)).squeeze(1)      # logit of P(low quality)
class CNNGate:
    def __init__(self, net, window, mean, std): self.net, self.window, self.mean, self.std = net.eval(), int(window), np.asarray(mean, np.float32), np.asarray(std, np.float32)
    def run_reference(self, windows):
        """median of the raw scalar features over the first 500 reads of a run (windows: >=1 array of shape (n, window) in arrival order)"""
        return np.median(scalars(np.asarray(windows)[:500, :self.window]), axis=0)
    def score(self, windows, ref, kit="native", batch=1024):
        """P(pass) for each row of `windows` (n, >=window samples in pA)"""
        w = np.asarray(windows)[:, :self.window]; X = normalize(w); S = (relative(scalars(w), ref) - self.mean) / self.std
        K = np.tile(np.eye(3, dtype=np.float32)[KITS.get(kit, kit)], (len(w), 1)); SC = np.hstack([S, K]).astype(np.float32); out = []
        with torch.no_grad():
            for i in range(0, len(w), batch): out.append(torch.sigmoid(self.net(torch.from_numpy(X[i:i+batch].astype(np.float32))[:, None], torch.from_numpy(SC[i:i+batch]))).numpy())
        return 1 - np.concatenate(out)
def save_cnn_gate(net, window, mean, std, path, notes=""):
    torch.save({"state": net.state_dict(), "window": int(window), "mean": np.asarray(mean, np.float32), "std": np.asarray(std, np.float32), "notes": notes}, path)
def load_cnn_gate(path):
    d = torch.load(path, map_location="cpu", weights_only=False); net = Net(len(d["mean"]) + 3); net.load_state_dict(d["state"]); return CNNGate(net, d["window"], d["mean"], d["std"])
