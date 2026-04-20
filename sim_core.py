# src/sim_core.py
import simpy, numpy as np, yaml, pandas as pd
from dataclasses import dataclass
from pathlib import Path
from policy_ml import build_candidates, choose_job

rngs = {}  # Separate RNG streams
def set_seeds(base_seed=42):
    global rngs
    rngs = {
        "arrival": np.random.default_rng(base_seed + 1),
        "service": np.random.default_rng(base_seed + 2),
        "routing": np.random.default_rng(base_seed + 3),
        "failures": np.random.default_rng(base_seed + 4),
   
    }

def gamma_from_mean_cv(mean, cv, size=None, rng=None):
    k = 1.0/(cv**2)          # shape
    theta = mean/k           # scale
    return rng.gamma(shape=k, scale=theta, size=size)

def atc_score(p_i, slack, tau, k):
    return np.exp(-max(0.0, slack)/(k * max(tau, 1e-9))) / max(p_i, 1e-9)

# ---------- Data Structures ----------
@dataclass
class Job:
    jid: int
    r_time: float
    due: float
    p: list                  # [p1,p2,p3]
    op: int = 0
    release_time: float | None = None
    start_times: list = None
    end_times: list = None
    def __post_init__(self):
        self.start_times, self.end_times = [], []

# ---------- Resources ----------
class Machine:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.res = simpy.Resource(env, capacity=1)
        self.queue = []
        self.current_job = None
        self.current_start = None
        self.current_ptime = None
        self.is_up = True                
        self.proc_handle = None           

class WLCGate:
    def __init__(self, cap):
        self.cap = cap
        self.on_floor = 0
    def can_release(self): return self.on_floor < self.cap
    def on_release(self):  self.on_floor += 1
    def on_depart(self):   self.on_floor = max(0, self.on_floor - 1)

# ---------- Shop ----------
class FlowShop3:
    # ---- Auto-Cap Helper ----
    def _compute_means_for_cap(self):
        m1, m2, m3 = map(float, self.cfg["processing_times"]["mean"])
        return m1, m2, m3

    def _auto_cap(self):
        """Cap = beta * E[P] / p_BN"""
        cap_cfg = self.cfg.get("wlc", {}).get("cap", {})
        beta = float(cap_cfg.get("beta", 3.5))
        m1, m2, m3 = self._compute_means_for_cap()
        means = [m1, m2, m3]
        p_bn = max(means)               # Bottleneck = highest mean
        E_P  = sum(means)               # E[P] = m1+m2+m3
        cap  = int(round(beta * (E_P / p_bn)))
        if "min" in cap_cfg: cap = max(int(cap_cfg["min"]), cap)
        if "max" in cap_cfg: cap = min(int(cap_cfg["max"]), cap)
      
        self._cap_auto_meta = {
            "beta": beta, "m1": m1, "m2": m2, "m3": m3,
            "E_P": E_P, "p_BN": p_bn, "BN": 1 + int(means.index(p_bn)),
            "cap": cap
        }
        return cap
    
    def _bn_index(self):
        """Bottleneck index based on average values (0/1/2)."""
        m1, m2, m3 = self._compute_means_for_cap()
        means = [m1, m2, m3]
        return int(np.argmax(means))


    def _workload_cap_minutes(self):
        norm = self.cfg.get("wlc", {}).get("norm", {})
        K = float(norm.get("K", norm.get("beta", 4.5)))  # K = Scenario lever
        m1, m2, m3 = self._compute_means_for_cap()
        p_bn = max([m1, m2, m3])
        m_bn = int(self.cfg["shop"].get("bn_servers", 1))
        wl_cap = K * p_bn * m_bn          # (Minute cap at BN)
        return wl_cap

    def _bn_workload_minutes(self):
        """
        BN workload in minutes:
        - All queued jobs still pending on BN: + p_bn
        - Job currently running on BN: + remaining time on BN
        """
        bn = self._bn_index()
        wl = 0.0
        # Queues
        for m in self.M:
            for j in m.queue:
                if (j.release_time is not None) and (j not in self.jobs_done) and (j.op <= bn):
                    wl += j.p[bn]
        # Job currently running on BN
        m_bn = self.M[bn]
        cj = m_bn.current_job
        if cj is not None and cj.op == bn:
            elapsed = self.env.now - (m_bn.current_start or self.env.now)
            remaining = max((m_bn.current_ptime or 0.0) - max(elapsed, 0.0), 0.0)
            wl += remaining
        # Current jobs before BN
        for k in range(bn):
            m_up = self.M[k]
            cj = m_up.current_job
            if cj is not None and cj.op <= bn:
                wl += cj.p[bn]
        return wl

    def __init__(self, cfg):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.M = [Machine(self.env, f"M{i+1}") for i in range(3)]

        # --- Determine Active WLC and Cap ---
        wlc_cfg = cfg.get("wlc", {})
        self.wlc_enabled = bool(wlc_cfg.get("enabled", True))
        cap_field = wlc_cfg.get("cap", {})

        if isinstance(cap_field, dict) and cap_field.get("mode") == "auto":
            cap = self._auto_cap()  # beta * E[P] / p_BN
        elif isinstance(cap_field, dict) and "value" in cap_field:
            cap = int(cap_field["value"])
        elif isinstance(cap_field, (int, float)):
            cap = int(cap_field)
        else:
            # Backward compatible (rho_0_80/95)
            arr = cfg.get("arrival", {})
            rho_list = arr.get("rho_levels", [0.80])
            rho = rho_list[0] if isinstance(rho_list, (list, tuple)) and rho_list else 0.80
            cf = cap_field if isinstance(cap_field, dict) else {}
            cap = int(cf.get("rho_0_80", 12)) if abs(rho - 0.80) < 0.1 else int(cf.get("rho_0_95", 18))

        if not self.wlc_enabled:
            cap = 10**9  # practically infinite (WLC off)

        self.wlc = WLCGate(cap)
        self._wip_timeline = [(0.0, 0)]  # (time, on_floor)

        # --- WLC / Pool-Logging ---
        self.wlc_blocks = 0                   # how often a release fails at the cap
        self._pool_timeline = [(0.0, 0)]      # (time, jobs_pool)

        # Logs / Settings
        self.jobs_pool, self.jobs_done = [], []
        self.logs = {"job": [], "events": [], "scores": []}
        # --- Störungs-KPIs sammeln ---
        self._fail_counts   = [0, 0, 0]
        self._down_start    = [None, None, None]   # Start time of a running down per machine
        self._down_minutes  = [0.0, 0.0, 0.0]      # Cumulative downtime per machine
        self._interrupts    = [0, 0, 0]            # how often the job was actually interrupted
        self._last_fail_end = [0.0, 0.0, 0.0]  # Time of the last FAIL_END per machine

        # ML / Decision-Logging
        self.decision_id = 0
        self.logs["ml_rows"] = []

        # Machine busy time (for utilization)
        self._busy_time = [0.0, 0.0, 0.0]

        self.verbose = bool(self.cfg.get("logging", {}).get("verbose", False))
        self.max_print = int(self.cfg.get("logging", {}).get("max_print", 0))
        self._print_count = 0

      # ATC parameter (reads k OR k_grid[0])
        atc_cfg = cfg.get("policies", {}).get("atc", {})
        if "k" in atc_cfg:
            self.atc_k = float(atc_cfg["k"])
        elif "k_grid" in atc_cfg:
            self.atc_k = float(atc_cfg["k_grid"][0])
        else:
            self.atc_k = 2.0  # Default

        self.logs.setdefault("meta", [{}])
        self.logs["meta"][0]["atc_k"] = float(self.atc_k)


        # --- Arrival process by mode ---
        mode = self.cfg.get("mode", {}).get("run", "generator")
        self.env.process(self.arrivals_poisson())

        # Machine processes
        for i in range(3):
            self.env.process(self.machine_runner(i))

         # Periodic top-up of the WLC gate (neutral, prevents zero queues)
        self.env.process(self.wlc_topup(
            period=self.cfg.get("wlc", {}).get("topup_period", 2.0)
        ))

  
        # --- Failure rates per machine ---
        dist_cfg = self.cfg.get("disturbances", {})
        self.dist_enabled = bool(dist_cfg.get("enabled", False))
        self.scen_dist = str(self.cfg.get("scenario", {}).get("dist", "off")).lower()

        if self.dist_enabled and self.scen_dist in ("low", "high"):
            self.bkd_cfg = dist_cfg.get("breakdowns", {})
            for i in range(3):
                self.env.process(self.failure_process(i))
        else:
            self.bkd_cfg = {}

   
        # Display Auto-Cap information directly
        if hasattr(self, "_cap_auto_meta") and self.verbose:
            meta = self._cap_auto_meta
            print(
                f"[AUTO-CAP] m1={meta['m1']:.1f}, m2={meta['m2']:.1f}, m3={meta['m3']:.1f}, "
                f"E[P]={meta['E_P']:.1f}, BN=M{meta['BN']} (p_BN={meta['p_BN']:.1f}), "
                f"beta={meta['beta']:.2f} -> Cap={meta['cap']}"
            )
        
        self.warmup_min = 0.0


    def failure_process(self, idx):
        """
        Causes periodic failures on machine idx:
        - Set TTF to ~ Exp(MTTF), set is_up=False, log FAIL_START, interrupt if job is running
        - Repair (MTTR), then set is_up=True, log FAIL_END
        """
        m = self.M[idx]
        level = self.scen_dist  # "low" or "high"
        mttf = float(self.bkd_cfg[level]["mttf"])
        mttr = float(self.bkd_cfg[level]["mttr"])

        while True:
            # Time until the next disturbance
            ttf = rngs["failures"].exponential(mttf)
            yield self.env.timeout(ttf)

            # disturbance begins
            self._fail_counts[idx] += 1
            m.is_up = False
            self._down_start[idx] = self.env.now
            if m.current_job is not None:
                self._interrupts[idx] += 1
            self.logs["events"].append({"time": self.env.now, "event": "FAIL_START",
                                        "machine": idx+1,
                                        "job": (m.current_job.jid if m.current_job else None)})
            
            # Interrupt the current job
            if m.proc_handle is not None and getattr(m.proc_handle, "alive", True):
                try:
                    m.proc_handle.interrupt()
                except Exception:
                    pass  # if it's already done

            # Repair
            ttr = rngs["failures"].exponential(mttr)
            yield self.env.timeout(ttr)

            # The disturbance ends
            m.is_up = True
            # Add downtime
            if self._down_start[idx] is not None:
                self._down_minutes[idx] += (self.env.now - self._down_start[idx])
                self._down_start[idx] = None
            self._last_fail_end[idx] = self.env.now  # Zeitpunkt merken
            
            self.logs["events"].append({"time": self.env.now, "event": "FAIL_END",
                                    "machine": idx+1})


    # ----- Arrival: Generator (Poisson + Gamma-PTs) -----
    def arrivals_poisson(self):
        means = self.cfg["processing_times"]["mean"]         # [m1,m2,m3]
        cv = self.cfg["processing_times"].get("cv", [0.5,0.5,0.5])
        if isinstance(cv, (int, float)):
            cv = [cv, cv, cv]

        rho = self.cfg["arrival"]["rho_levels"][0]           

        # --- determine BN & λ ---
        bn   = self.cfg["shop"]["bottleneck_machine"] - 1   # Index machine
        p_bn = float(self.cfg["processing_times"]["mean"][bn])
        m_bn = int(self.cfg["shop"].get("bn_servers", 1))
        lam_per_min = (rho * m_bn) / p_bn  # Jobs/minute

        # --- Meta-Log ---
        wl_cap = self._workload_cap_minutes()  # Cap delivers in minutes
        self.logs.setdefault("meta", []).append({
            "rho_set": rho,
            "bn_index": bn,
            "p_bn_mean": p_bn,
            "m_bn": m_bn,
            "lambda_per_min": lam_per_min,
            "wlc_cap_minutes": wl_cap
        })
        print(f"[INIT] ρ={rho:.2f}, λ={lam_per_min:.4f}/min, BN μ={p_bn:.2f} min, m_bn={m_bn}, Cap={wl_cap:.1f} min")


        jid = 0
        while True:
            dt = rngs["arrival"].exponential(1.0/lam_per_min)
            yield self.env.timeout(dt)
            p1 = float(gamma_from_mean_cv(means[0], cv[0], rng=rngs["service"]))
            p2 = float(gamma_from_mean_cv(means[1], cv[1], rng=rngs["service"]))
            p3 = float(gamma_from_mean_cv(means[2], cv[2], rng=rngs["service"]))
            r = self.env.now
            Ptot = p1+p2+p3
            dd_key = self.cfg.get("scenario", {}).get("due", "mittel")
            alpha  = float(self.cfg["duedate"]["tightness"][dd_key])
            due = r + alpha * Ptot
            jit = self.cfg["duedate"].get("jitter_frac", 0.0)
            jit_min = self.cfg["duedate"].get("jitter_min", 0.0)
            if jit > 0.0:
                amp = max(jit, jit_min)
                factor = 1.0 + rngs["routing"].uniform(-amp, amp)
                due = r + max(0.0, (due - r) * factor)

            job = Job(jid=jid, r_time=r, due=due, p=[p1,p2,p3])
            self.jobs_pool.append(job)
            self._note_pool_wip()
            self.logs["events"].append({"time": self.env.now, "event": "arrival", "job": jid})
            jid += 1
            self.try_release_from_pool()
    
    def _note_wip(self):
        """Record time/level when on_floor changes."""
        self._wip_timeline.append((self.env.now, self.wlc.on_floor))
        
    def _note_pool_wip(self):
        """Record time/level when the pool size changes."""
        self._pool_timeline.append((self.env.now, len(self.jobs_pool)))

    def wlc_topup(self, period=2.0):
        """Periodic replenishment through the gate – keeps queues stable without predetermining the sequence."""
        while True:
            yield self.env.timeout(period)
            self.try_release_from_pool()

    
    # ----- WLC-Approval -----
    def try_release_from_pool(self):
        """
        WLC-Release: Standard = neutral (FIFO).
        wlc.mode == "workload": Available until WL(BN) < WL_cap (minutes).
        """
        mode = self.cfg.get("wlc", {}).get("mode", None)
        wl_cap = self._workload_cap_minutes() if mode == "workload" else None

        def can_release_more():
            if not self.wlc_enabled:
                return True
            if wl_cap is None:
                # Fallback
                return self.wlc.can_release()
            # Workload-Norm (BN-Minutes)
            return self._bn_workload_minutes() < wl_cap

        while can_release_more() and self.jobs_pool:
            t = self.env.now

            # Pool rule: keep neutral (FIFO) for fair heuristic benchmarks
            pool_rule = (
                self.cfg.get("wlc", {}).get("cap", {}).get("pool_rule") or
                self.cfg.get("wlc", {}).get("pool_rule") or
                "FIFO"
            )
            dispatch_rule = self.cfg.get("policies", {}).get("heuristics", ["SPT"])[0]

            def pick_fifo():
                return self.jobs_pool[0]

            def pick_min_slack():
                def rest_from_op(j): return sum(j.p[j.op:])
                return min(self.jobs_pool, key=lambda jj: jj.due - t - rest_from_op(jj))

            def pick_align():
                # OPTIONAL
                if dispatch_rule == "SPT":
                    return min(self.jobs_pool, key=lambda jj: jj.p[jj.op])
                elif dispatch_rule == "EDD":
                    return min(self.jobs_pool, key=lambda jj: jj.due)
                elif dispatch_rule == "ATC":
                    tau = (np.mean([jj.p[jj.op] for jj in self.jobs_pool])
                        if self.jobs_pool else
                        self.cfg.get("processing_times", {}).get("mean", [10,10,10])[0])
                    k = self.atc_k
                    def atc_s(jj):
                        s = jj.due - t - sum(jj.p[jj.op:])
                        return np.exp(-max(0.0, s)/(k*max(tau, 1e-9))) / max(jj.p[jj.op], 1e-9)
                    return max(self.jobs_pool, key=atc_s)
                else:
                    return pick_min_slack()

            if pool_rule.upper() == "FIFO":
                j = pick_fifo()
            elif pool_rule.upper() == "ALIGN":
                j = pick_align()
            elif pool_rule.upper() == "EDD":
                j = min(self.jobs_pool, key=lambda jj: jj.due)
            elif pool_rule.upper() == "SPT":
                j = min(self.jobs_pool, key=lambda jj: jj.p[jj.op])
            else:
                j = pick_min_slack()

            # Share
            self.jobs_pool.remove(j)
            self._note_pool_wip()
            self.wlc.on_release()
            self._note_wip()
            j.release_time = t
            self.logs["events"].append({"time": t, "event": "release", "job": j.jid})
            self.M[0].queue.append(j)

            if wl_cap is not None and self._bn_workload_minutes() >= wl_cap:
                break
        
        # Finally: if there are still jobs in the pool but the cap is full → count the block
        if self.wlc_enabled and wl_cap is not None and self.jobs_pool and (not can_release_more()):
            self.wlc_blocks += 1
            self.logs["events"].append({
                "time": self.env.now,
                "event": "wlc_block",
                "pool_size": len(self.jobs_pool),
                "bn_workload_now": self._bn_workload_minutes()
            })

   
    # ----- Dispatching (SPT/EDD/ATC) + Trace -----
    def dispatch(self, idx):
        rule = self.cfg["policies"]["heuristics"][0]
        q, t = self.M[idx].queue, self.env.now

        def rest_from_i(j): 
            return sum(j.p[idx:])

        # List of candidates for Logging
        cand = []
        for j in q:
            cand.append({"job": j.jid, "p_i": j.p[idx], "due": j.due,
                        "slack": j.due - t - rest_from_i(j)})

        # -------- Selection by rule --------
        if rule == "SPT":
            selected = min(q, key=lambda j: j.p[idx])
            for c in cand: 
                c["score"] = -c["p_i"]

        elif rule == "EDD":
            selected = min(q, key=lambda j: j.due)
            for c in cand: 
                c["score"] = -c["due"]

        elif rule == "ATC":
            tau_local = np.mean([jj.p[idx] for jj in q]) if q else self.cfg["processing_times"]["mean"][idx]
            k = self.atc_k
            scores = []
            for j in q:
                slack = j.due - t - sum(j.p[idx:])
                scores.append(atc_score(j.p[idx], slack, tau_local, k))
            sel_idx = int(np.argmax(scores))
            selected = q[sel_idx]
            for i, c in enumerate(cand):
                c["score"] = float(scores[i])

        elif rule == "ML":
            scenario_state = {
                "scenario_due": str(self.cfg.get("scenario", {}).get("due", "")),
                "scenario_dist": str(self.cfg.get("scenario", {}).get("dist", "off")),
                "rho_set": float(
                    self.logs.get("meta", [{}])[0].get(
                        "rho_set",
                        self.cfg.get("arrival", {}).get("rho_levels", [0.0])[0]
                    )
                ),
                "K": float(self.cfg.get("wlc", {}).get("norm", {}).get("K", 0.0)),
                "bn_workload_now": float(self._bn_workload_minutes()),
            }

            waiting_jobs = []
            for jj in q:
                rest  = sum(jj.p[idx:])
                slack = jj.due - t - rest
                age   = t - (jj.release_time if jj.release_time is not None else jj.r_time)

                waiting_jobs.append({
                    "job_id": jj.jid,
                    "machine": idx + 1,
                    "p_i": jj.p[idx],
                    "p_i_rest_from_i": rest,
                    "slack": slack,
                    "age": age,
                    "queue_len_m_on_floor": len(q),
                    "on_floor": self.wlc.on_floor if hasattr(self, "wlc") else 0,
                    "wip_cap": self.wlc.cap if hasattr(self, "wlc") else 0.0,
                    "tau": self.cfg.get("processing_times", {}).get("mean", [10,10,10])[idx],
                })

            dfC = build_candidates(waiting_jobs, now=t, scenario_state=scenario_state)

            sel_job_id, dfC_scored = choose_job("models/model_rct.pkl", dfC)

            score_map = dict(zip(dfC_scored["job_id"], dfC_scored["ml_score"]))
            selected = next(jj for jj in q if jj.jid == sel_job_id)

            for c in cand:
                c["score"] = float(score_map.get(c["job"], float("nan")))


        
        # -------- ML dataset per candidate --------
        decision_id = self.decision_id; self.decision_id += 1
        q_len = len(q)
        on_floor = self.wlc.on_floor
        wip_cap = self.wlc.cap
        tau_for_log = (np.mean([jj.p[idx] for jj in q]) if q else
                    self.cfg.get("processing_times", {}).get("mean", [10,10,10])[idx])
        policy_str = rule
        t_now = self.env.now

        level = str(getattr(self, "scen_dist", "off"))
        mttf_cfg = float(self.bkd_cfg[level]["mttf"]) if (level in ("low","high") and self.bkd_cfg) else 0.0
        mttr_cfg = float(self.bkd_cfg[level]["mttr"]) if (level in ("low","high") and self.bkd_cfg) else 0.0
        avail = self.machine_availability_est(idx)
        tslf  = self.time_since_last_fail(idx)
        bn_wl = self._bn_workload_minutes()

        for jj in q:
            rest = sum(jj.p[idx:])
            slack = jj.due - t_now - rest
            age = t_now - (jj.release_time if jj.release_time is not None else jj.r_time)
            self.logs["ml_rows"].append({
                "decision_id": decision_id,
                "time": t_now,
                "machine": idx+1,
                "policy": policy_str,
                "job": jj.jid,
                "selected": 1 if jj is selected else 0,
                "p_i": jj.p[idx],
                "rest_from_i": rest,
                "due_minus_now": jj.due - t_now,
                "slack": slack,
                "age": age,
                "op": jj.op,
                "queue_len_m": q_len,
                "on_floor": on_floor,
                "wip_cap": wip_cap,
                "tau": tau_for_log,
                # --- Scenarios & Disturbances ---
                "scenario_due": str(self.cfg.get("scenario", {}).get("due", "")),
                "scenario_dist": level,
                "rho_set": float(self.logs.get("meta",[{}])[0].get("rho_set",
                            self.cfg.get("arrival",{}).get("rho_levels",[0.0])[0])),
                "K": float(self.cfg.get("wlc",{}).get("norm",{}).get("K", 0.0)),
                "bn_workload_now": bn_wl,
                "machine_is_up": 1 if self.M[idx].is_up else 0,
                "avail_est": avail,
                "time_since_last_fail": tslf,
                "mttf_cfg": mttf_cfg,
                "mttr_cfg": mttr_cfg,
                "fails_so_far_m": int(self._fail_counts[idx]),
                "down_minutes_m": float(self._down_minutes[idx]),
            })

        # -------- Trace/Logging --------
        self.logs["scores"].append({
            "time": t, "machine": idx+1, "policy": policy_str,
            "selected_job": selected.jid, "candidates": cand
        })

        if self.verbose and self._print_count < self.max_print:
            self._print_count += 1
            def fmt(c):
                s = c.get("score", float("nan"))
                return f"J{c['job']}(p={c['p_i']}, slack={c['slack']:.1f}, score={s:.3g})"
            print(f"[t={t:6.1f}] M{idx+1} {policy_str}: " + ", ".join(fmt(c) for c in cand)
                + f"  -> pick J{selected.jid}")
        return selected


    # ----- Machines -----
    def machine_runner(self, idx):
        m = self.M[idx]
        while True:
            while not m.queue:
                yield self.env.timeout(0.1)
            j = self.dispatch(idx)
            m.queue.remove(j)
            with m.res.request() as req:
                yield req
                j.start_times.append(self.env.now)
                ptime = j.p[idx]
                
                m.current_job = j
                m.current_start = self.env.now
                m.current_ptime = ptime
                
                self.logs["events"].append({"time": self.env.now, "event": "start",
                                            "machine": idx+1, "job": j.jid, "p": ptime})
                
                
                remaining = ptime
                proc_busy = 0.0  # actual processing time (excluding downtime)

                while remaining > 1e-9:
                    # If the machine is down, wait until it's back up
                    while not m.is_up:
                        yield self.env.timeout(0.1)

                    try:
                        m.proc_handle = self.env.process(self._hold_for(remaining))
                        t_before = self.env.now
                        yield m.proc_handle
                        delta = self.env.now - t_before
                        proc_busy += delta
                        remaining = 0.0
                    
                        # Count “busy” only after warm-up in _busy_time
                        start_seg = max(t_before, self.warmup_min)
                        end_seg   = self.env.now
                        if end_seg > start_seg:
                            self._busy_time[idx] += (end_seg - start_seg)
                    
                    except simpy.Interrupt:
                        # interrupted by FAIL_START
                        delta = self.env.now - t_before
                        proc_busy += max(delta, 0.0)
                        remaining = max(remaining - delta, 0.0)
                        
                        # Here, too, count “Busy” only after the warm-up
                        start_seg = max(t_before, self.warmup_min)
                        end_seg   = self.env.now
                        if end_seg > start_seg:
                            self._busy_time[idx] += (end_seg - start_seg)
                        
                        # wait until repaired (m.is_up becomes True in the failure_process)
                        while not m.is_up:
                            yield self.env.timeout(0.1)
                        # loop continues

                j.end_times.append(self.env.now)
                self.logs["events"].append({"time": self.env.now, "event": "end",
                                            "machine": idx+1, "job": j.jid})
                # Clean-up
                m.current_job = None
                m.current_start = None
                m.current_ptime = None
                m.proc_handle = None
                
            
            j.op += 1
            if j.op < 3:
                self.M[idx+1].queue.append(j)
            else:
                self.jobs_done.append(j)
                self.wlc.on_depart()
                self._note_wip()
                self.log_job(j)
                self.try_release_from_pool()

    
    def _hold_for(self, dt):
        yield self.env.timeout(dt)
    
    
    def log_job(self, j: Job):
        self.logs["job"].append({
            "job_id": j.jid,
            "release": j.release_time,
            "completion": j.end_times[-1],
            "due": j.due,
            "tardiness": max(0.0, j.end_times[-1]-j.due),
            "on_time": 1.0 if j.end_times[-1] <= j.due else 0.0,
            "ct": j.end_times[-1] - j.release_time,
        })

    def machine_availability_est(self, idx):
        down = self._down_minutes[idx]
        if self._down_start[idx] is not None:
            down += (self.env.now - self._down_start[idx])  
        sim_t = max(self.env.now, 1e-9)
        return max(0.0, 1.0 - down / sim_t)

    def time_since_last_fail(self, idx):
        last = self._last_fail_end[idx]
        return self.env.now - last if last > 0 else 0.0

    def run(self, horizon_h=50, warmup_h=10):
        self.warmup_min = warmup_h * 60
        self.env.run(until=horizon_h*60)
        
        bn = self._bn_index()
        m_bn = int(self.cfg["shop"].get("bn_servers", 1))
        sim_t = max(1e-9, self.env.now)
        busy_bn = self._busy_time[bn]                
        rho_realized = busy_bn / (sim_t * m_bn)
        self.logs.setdefault("meta", []).append({"rho_realized_bn": rho_realized})
        print(f"[END] realized ρ_BN = {rho_realized:.3f}")

        # --- Export-Logs ---
        out_dir = Path(self.cfg.get("logging", {}).get("out_dir", "out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        df_job = pd.DataFrame(self.logs["job"])
        df_evt = pd.DataFrame(self.logs["events"])

        rows = []
        for rec in self.logs["scores"]:
            t = rec["time"]; m = rec["machine"]; pol = rec["policy"]; sel = rec["selected_job"]
            for c in rec["candidates"]:
                rows.append({
                    "time": t, "machine": m, "policy": pol, "selected_job": sel,
                    "job": c["job"], "p_i": c["p_i"], "slack": c["slack"],
                    "score": c.get("score", float("nan"))
                })
        df_sc = pd.DataFrame(rows)

        if not df_job.empty: df_job.to_csv(out_dir / "job_log.csv", index=False)
        if not df_evt.empty: df_evt.to_csv(out_dir / "event_log.csv", index=False)
        if not df_sc.empty:  df_sc.to_csv(out_dir / "policy_scores.csv", index=False)
        # Export ML dataset
        df_ml = pd.DataFrame(self.logs.get("ml_rows", []))
        if not df_ml.empty:
            df_ml.to_csv(out_dir / "ml_dataset.csv", index=False)

        if df_job.empty:
            return {"n": 0, "on_time_rate": 0.0, "ct_mean": 0.0, "wip_avg": 0.0, "wip_max": 0}

        df = df_job[df_job["release"] >= warmup_h*60]

        # --- WIP (time-weighted average) ---
        def _avg_wip(timeline, t0, t1):
            if timeline[-1][0] < t1:
                timeline = timeline + [(t1, timeline[-1][1])]
            area = 0.0
            for (ta, wa), (tb, wb) in zip(timeline, timeline[1:]):
                if tb <= t0 or ta >= t1:
                    continue
                a = max(ta, t0); b = min(tb, t1)
                area += (b - a) * wa
            return area / max(t1 - t0, 1e-9)

        T0 = warmup_h * 60
        T1 = horizon_h * 60
        wip_avg = _avg_wip(self._wip_timeline, T0, T1)
        wip_max = max(w for t, w in self._wip_timeline if t >= T0) if self._wip_timeline else 0
        wip_pool_avg = _avg_wip(self._pool_timeline, T0, T1)
        wip_pool_max = max(w for t, w in self._pool_timeline if t >= T0) if self._pool_timeline else 0

        
        # --- WIP only during active time (on_floor > 0) ---
        def _busy_stats(timeline, t0, t1):
            if not timeline:
                return 0.0, 0.0
            # Fit the timeline to the window
            if timeline[0][0] > t0:
                timeline = [(t0, timeline[0][1])] + timeline
            if timeline[-1][0] < t1:
                timeline = timeline + [(t1, timeline[-1][1])]
            area_busy = 0.0
            busy_time = 0.0
            for (ta, wa), (tb, wb) in zip(timeline, timeline[1:]):
                a = max(ta, t0); b = min(tb, t1)
                if b <= a:
                    continue
                if wa > 0:  # Only active sections count
                    area_busy += (b - a) * wa
                    busy_time += (b - a)
            wip_avg_busy = area_busy / max(busy_time, 1e-9)
            return wip_avg_busy, busy_time

        wip_avg_busy, busy_minutes = _busy_stats(self._wip_timeline, T0, T1)

        # optional KPIs
        eff_minutes = max(T1 - T0, 1e-9)
        n_eff = len(df)
        throughput_per_min = n_eff / eff_minutes if eff_minutes > 0 else 0.0

        # percentile
        ct_p50 = float(df["ct"].median()) if n_eff > 0 else 0.0
        ct_p95 = float(df["ct"].quantile(0.95)) if n_eff > 0 else 0.0
        tard_p95 = float(df["tardiness"].quantile(0.95)) if n_eff > 0 else 0.0

        # Utilization per machine (busy time / window)
        util_m = [bt / eff_minutes for bt in self._busy_time]

        # --- disturbance KPIs ---
        sim_t = max(1e-9, self.env.now)
        fails_m = self._fail_counts
        down_m  = self._down_minutes
        avail_m = [max(0.0, 1.0 - (down / sim_t)) for down in down_m]
        mttr_real = [(down / f) if f > 0 else 0.0 for down, f in zip(down_m, fails_m)]
      
        uptime_m = [sim_t - down for down in down_m]
        mttf_real = [(up / f) if f > 0 else 0.0 for up, f in zip(uptime_m, fails_m)]
        fails_total = int(sum(fails_m))


        return {
            "n": int(len(df)),
            "on_time_rate": float(df["on_time"].mean()) if len(df)>0 else 0.0,
            "ct_mean": float(df["ct"].mean()) if len(df)>0 else 0.0,
            "wip_avg": float(wip_avg),
            "wip_avg_busy": float(wip_avg_busy),
            "busy_minutes": float(busy_minutes),
            "wip_max": int(wip_max),
            "wip_pool_avg": float(wip_pool_avg),
            "wip_pool_max": int(wip_pool_max),
            "wlc_blocks": int(self.wlc_blocks),
            "throughput_per_min": float(throughput_per_min),
            "ct_p50": float(ct_p50),
            "ct_p95": float(ct_p95),
            "tardiness_p95": float(tard_p95),
            "util_m1": float(util_m[0]),
            "util_m2": float(util_m[1]),
            "util_m3": float(util_m[2]),
            "dist_level": str(getattr(self, "scen_dist", "off")),
            "fails_total": fails_total,
            "fails_m1": int(fails_m[0]), "fails_m2": int(fails_m[1]), "fails_m3": int(fails_m[2]),
            "down_m1_min": float(down_m[0]), "down_m2_min": float(down_m[1]), "down_m3_min": float(down_m[2]),
            "avail_m1": float(avail_m[0]), "avail_m2": float(avail_m[1]), "avail_m3": float(avail_m[2]),
            "mttr_m1": float(mttr_real[0]), "mttr_m2": float(mttr_real[1]), "mttr_m3": float(mttr_real[2]),
            "mttf_m1": float(mttf_real[0]), "mttf_m2": float(mttf_real[1]), "mttf_m3": float(mttf_real[2]),
            "interrupts_m1": int(self._interrupts[0]),
            "interrupts_m2": int(self._interrupts[1]),
            "interrupts_m3": int(self._interrupts[2]),        
        }



# ---------- Runner ----------
def run_pilot(cfg_path="cfg/base.yaml", seed=42):
    from datetime import datetime

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    set_seeds(seed)

    pt = cfg.get("processing_times", {})
    cv = pt.get("cv", None)
    if cv is not None and isinstance(cv, (int, float)):
        pt["cv"] = [cv, cv, cv]
        cfg["processing_times"] = pt

    sim_cfg = cfg.get("sim", {})
    horizon_h = sim_cfg.get("horizon_hours", 50)
    warmup_h = sim_cfg.get("warmup_hours", 10)

    # build shop 
    shop = FlowShop3(cfg)

    # ------- A unique output directory for each run --------
    out_root = cfg.get("logging", {}).get("out_root", "runs")
    rho = float(cfg.get("arrival", {}).get("rho_levels", [0.95])[0])
    due = str(cfg.get("scenario", {}).get("due", "mittel"))
    K = float(cfg.get("wlc", {}).get("norm", {}).get("K",
        cfg.get("wlc", {}).get("norm", {}).get("beta", 4.5)))
    dist = str(cfg.get("scenario", {}).get("dist", "off"))
    cap_bn_min = int(round(shop._workload_cap_minutes()))

    ds = f"util{int(round(rho*100))}_due{due}_K{K:g}_dist{dist}"

    policy = cfg.get("policies", {}).get("heuristics", ["SPT"])[0]
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clear Cap folder name in minutes at BN
    run_dir = (Path(out_root) / ds / policy /
                f"capBN{cap_bn_min}min" / f"seed{seed}" / run_id)

    run_dir.mkdir(parents=True, exist_ok=True)

    shop.cfg.setdefault("logging", {})["out_dir"] = str(run_dir)
    print(f"[OUT] Writing results to: {run_dir}")
  
    res = shop.run(horizon_h=horizon_h, warmup_h=warmup_h)

    shop.logs.setdefault("meta", []).append({
    "rho_set": rho, "scenario_due": due, "scenario_K": K, "scenario_dist": dist,
    "cap_bn_minutes": cap_bn_min, "policy": policy, "seed": seed
    })

    import json, shutil

    (Path(run_dir) / "kpi_summary.json").write_text(json.dumps(res, indent=2))
    (Path(run_dir) / "cfg_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )

    latest = Path("out")
    latest.mkdir(exist_ok=True)
    shutil.copytree(run_dir, latest, dirs_exist_ok=True)  # overwrites files
    print(f"[OUT] Mirrored latest results to: {latest} (overwrite)")


    # Symlink 
    try:
        latest_link = Path(out_root) / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
        print(f"[OUT] Symlink created: {latest_link} -> {run_dir}")
    except Exception as e:
        # The fallback is the copy in ./out
        pass

    return res
