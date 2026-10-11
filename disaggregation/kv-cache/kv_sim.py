"""Multi-tier KV-cache simulator: eviction policies, tenant isolation, offload cost.

Why a simulator: on one Colab A100 we can't build a real HBM->CPU->remote tier
inside vLLM, but we CAN evaluate eviction / offload *policies* on realistic
block-hash traces and price them with transfer bandwidths measured by
kv_transfer_bench.py. Results are labelled "simulated" everywhere.

Model
-----
- Requests are sequences of block hashes (prefix chains, like vLLM's 16-token blocks).
- Tier 0 = HBM (capacity in blocks), Tier 1 = CPU DRAM (capacity in blocks).
- A block found in HBM costs 0 transfer; in CPU costs bytes/H2D_bandwidth;
  a miss costs recompute (prefill) time per block.
- Block hashes are namespaced by tenant salt when isolation=True, so tenants
  can never hit each other's blocks (mirrors vLLM cache_salt).

Policies: lru, lfu, tenant_fair (LRU, but evict from the tenant most over its fair share).
"""
from __future__ import annotations

import hashlib
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field


@dataclass
class CostModel:
    block_bytes: int  # bytes of K+V for one block across all layers
    h2d_gbps: float = 20.0  # measured pinned host->device GB/s (replace with kv_transfer_bench output)
    prefill_s_per_block: float = 1e-4  # measured recompute cost per block (from TTFT vs prompt length fit)

    def fetch_cpu_s(self) -> float:
        return self.block_bytes / (self.h2d_gbps * 1e9)


@dataclass
class TierStats:
    hbm_hits: int = 0
    cpu_hits: int = 0
    misses: int = 0
    evictions_hbm: int = 0
    evictions_cpu: int = 0
    transfer_s: float = 0.0
    recompute_s: float = 0.0
    per_tenant_hits: dict = field(default_factory=lambda: defaultdict(int))
    per_tenant_lookups: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def lookups(self) -> int:
        return self.hbm_hits + self.cpu_hits + self.misses

    def as_dict(self) -> dict:
        n = max(1, self.lookups)
        return {"lookups": self.lookups, "hbm_hit_rate": self.hbm_hits / n, "cpu_hit_rate": self.cpu_hits / n,
                "miss_rate": self.misses / n, "evictions_hbm": self.evictions_hbm,
                "evictions_cpu": self.evictions_cpu, "transfer_s": self.transfer_s,
                "recompute_s": self.recompute_s, "total_cost_s": self.transfer_s + self.recompute_s,
                "per_tenant_hit_rate": {t: self.per_tenant_hits[t] / max(1, self.per_tenant_lookups[t])
                                        for t in self.per_tenant_lookups}}


class Tier:
    """One cache tier with O(1) LRU, LFU (frequency buckets) and tenant-fair eviction."""

    def __init__(self, capacity: int, policy: str):
        if policy not in ("lru", "lfu", "tenant_fair"):
            raise ValueError(policy)
        self.capacity = capacity
        self.policy = policy
        self.blocks: OrderedDict[str, str] = OrderedDict()  # hash -> tenant, LRU order
        self.by_tenant: dict[str, OrderedDict] = defaultdict(OrderedDict)  # LRU order per tenant
        self.freq: dict[str, int] = {}
        self.buckets: dict[int, OrderedDict] = defaultdict(OrderedDict)  # freq -> hashes (LRU order)
        self.min_freq = 0

    def __len__(self) -> int:
        return len(self.blocks)

    def __contains__(self, h: str) -> bool:
        return h in self.blocks

    def owned(self, tenant: str) -> int:
        return len(self.by_tenant[tenant])

    def _bump(self, h: str) -> None:
        f = self.freq[h]
        del self.buckets[f][h]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq = f + 1
        self.freq[h] = f + 1
        self.buckets[f + 1][h] = None

    def touch(self, h: str) -> None:
        self.blocks.move_to_end(h)
        self.by_tenant[self.blocks[h]].move_to_end(h)
        self._bump(h)

    def insert(self, h: str, tenant: str) -> tuple[str, str] | None:
        """Insert; return the evicted (hash, tenant) if any."""
        if self.capacity <= 0:
            return (h, tenant)
        evicted = self._evict() if len(self.blocks) >= self.capacity else None
        self.blocks[h] = tenant
        self.by_tenant[tenant][h] = None
        self.freq[h] = 1
        self.buckets[1][h] = None
        self.min_freq = 1
        return evicted

    def remove(self, h: str) -> str:
        t = self.blocks.pop(h)
        del self.by_tenant[t][h]
        f = self.freq.pop(h)
        del self.buckets[f][h]
        if not self.buckets[f]:
            del self.buckets[f]
        return t

    def _evict(self) -> tuple[str, str]:
        if self.policy == "lru":
            h = next(iter(self.blocks))
        elif self.policy == "lfu":
            while self.min_freq not in self.buckets:
                self.min_freq += 1
            h = next(iter(self.buckets[self.min_freq]))
        else:  # tenant_fair: evict LRU block of the tenant most over its fair share
            active = [t for t, d in self.by_tenant.items() if d]
            fair = self.capacity / len(active)
            hog = max(active, key=lambda t: len(self.by_tenant[t]) - fair)
            h = next(iter(self.by_tenant[hog]))
        return h, self.remove(h)


class KVCacheSim:
    def __init__(self, hbm_blocks: int, cpu_blocks: int, policy: str, cost: CostModel, isolation: bool = True):
        self.hbm = Tier(hbm_blocks, policy)
        self.cpu = Tier(cpu_blocks, "lru")
        self.cost = cost
        self.isolation = isolation
        self.stats = TierStats()

    def _ns(self, h: str, tenant: str) -> str:
        return hashlib.sha1(f"{tenant}|{h}".encode()).hexdigest() if self.isolation else h

    def request(self, tenant: str, block_hashes: list[str]) -> dict:
        hit_prefix = True
        cost = 0.0
        for raw in block_hashes:
            h = self._ns(raw, tenant)
            self.stats.per_tenant_lookups[tenant] += 1
            if hit_prefix and h in self.hbm:
                self.hbm.touch(h)
                self.stats.hbm_hits += 1
                self.stats.per_tenant_hits[tenant] += 1
                continue
            if hit_prefix and h in self.cpu:
                self.cpu.remove(h)
                self.stats.cpu_hits += 1
                self.stats.per_tenant_hits[tenant] += 1
                c = self.cost.fetch_cpu_s()
                self.stats.transfer_s += c
                cost += c
                self._insert_hbm(h, tenant)
                continue
            hit_prefix = False  # prefix chain broken: everything after is recomputed
            self.stats.misses += 1
            self.stats.recompute_s += self.cost.prefill_s_per_block
            cost += self.cost.prefill_s_per_block
            self._insert_hbm(h, tenant)
        return {"cost_s": cost}

    def _insert_hbm(self, h: str, tenant: str) -> None:
        ev = self.hbm.insert(h, tenant)
        if ev is not None and ev[0] != h:
            self.stats.evictions_hbm += 1
            if self.cpu.capacity > 0:  # offload instead of drop
                self.stats.transfer_s += self.cost.fetch_cpu_s()  # D2H write-back (assume symmetric bandwidth)
                ev2 = self.cpu.insert(*ev)
                if ev2 is not None:
                    self.stats.evictions_cpu += 1


def synthetic_trace(n_requests: int = 1500, tenants=("acme", "globex"), n_prefixes: int = 24,
                    prefix_blocks: int = 128, suffix_blocks: int = 8, reuse: float = 0.8, zipf_s: float = 1.1,
                    hog: str | None = None, seed: int = 0) -> list[tuple[str, list[str]]]:
    """Zipf-popular shared prefixes per tenant + unique suffixes. `hog` tenant sends 3x traffic with unique prompts."""
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** zipf_s for i in range(n_prefixes)]
    trace = []
    for i in range(n_requests):
        t = rng.choice(list(tenants) + ([hog] * 3 if hog else []))
        if t != hog and rng.random() < reuse:
            p = rng.choices(range(n_prefixes), weights)[0]
            chain = [f"{t}-p{p}-b{j}" for j in range(prefix_blocks)]
        else:
            chain = [f"{t}-u{i}-b{j}" for j in range(prefix_blocks)]
        chain += [f"{t}-r{i}-s{j}" for j in range(suffix_blocks)]
        # make hashes prefix-chained like vLLM
        hashed, prev = [], ""
        for b in chain:
            prev = hashlib.sha1((prev + b).encode()).hexdigest()[:16]
            hashed.append(prev)
        trace.append((t, hashed))
    return trace


def run(trace, hbm_blocks, cpu_blocks, policy, cost, isolation=True) -> dict:
    sim = KVCacheSim(hbm_blocks, cpu_blocks, policy, cost, isolation)
    for t, chain in trace:
        sim.request(t, chain)
    return sim.stats.as_dict()


def sweep(cost: CostModel, out_path: str | None = None) -> dict:
    """Compare policies x HBM size x CPU offload on a fair and a 'noisy neighbour' trace."""
    import json

    results = {"label": "SIMULATED - policy comparison on synthetic traces, priced with measured costs",
               "cost_model": cost.__dict__, "runs": []}
    for trace_name, hog in (("fair", None), ("noisy_neighbour", "freetier")):
        tenants = ("acme", "globex") if hog is None else ("acme", "globex", "freetier")
        trace = synthetic_trace(tenants=tenants, hog=hog)
        for hbm in (512, 1024, 2048):
            for cpu in (0, 4096):
                for policy in ("lru", "lfu", "tenant_fair"):
                    r = run(trace, hbm, cpu, policy, cost)
                    results["runs"].append({"trace": trace_name, "hbm_blocks": hbm, "cpu_blocks": cpu,
                                            "policy": policy, **r})
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--block-bytes", type=int, default=16 * 2 * 28 * 2 * 128 * 2,
                    help="default: Qwen2.5-1.5B, 16 tok/block * K,V * 28 layers * 2 kv heads * 128 dim * 2 B")
    ap.add_argument("--h2d-gbps", type=float, default=20.0)
    ap.add_argument("--prefill-s-per-block", type=float, default=1e-4)
    ap.add_argument("--transfer-report", default=None, help="kv_transfer report.json to pull measured bandwidth")
    ap.add_argument("--out", default="kv_sim_results.json")
    a = ap.parse_args()
    h2d = a.h2d_gbps
    if a.transfer_report:
        rep = json.load(open(a.transfer_report))
        h2d = rep.get("bandwidth", {}).get("h2d_pinned_GBps", h2d)
    res = sweep(CostModel(a.block_bytes, h2d, a.prefill_s_per_block), a.out)
    best = sorted(res["runs"], key=lambda r: r["total_cost_s"])[:5]
    for r in best:
        print({k: r[k] for k in ("trace", "hbm_blocks", "cpu_blocks", "policy", "hbm_hit_rate", "total_cost_s")})
