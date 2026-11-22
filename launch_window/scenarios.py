"""条款比较、情景网格、风险边界与共享预留。

比较口径固定为同一 (延期天数, 通知提前量, 窗口是否已确认)，逐条产品在同一
实际窗口下核对，而不是让每条产品挑自己最能接住的日期。候选实际窗口互斥，
共享预留取各窗口需求的最大值；同一窗口内不同客群可能同时接受，需求相加。
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .allocate import Conflict, Plan, allocate, resource_keys
from .behavior import cohort_stats, outcome_probabilities
from .model import Booking, Dataset, Terms

MAX_DELAY = 3
MAX_LEAD = 4


@dataclass(frozen=True)
class Scenario:
    delay: int
    lead: int
    confirmed: bool = True
    label: str = ""

    @property
    def name(self) -> str:
        state = "已确认" if self.confirmed else "窗口未定"
        return self.label or f"延期{self.delay}天/提前{self.lead}天/{state}"


@dataclass
class ScenarioResult:
    scenario: Scenario
    terms: Terms
    plans: dict[str, Plan]
    refund_cap: int

    @property
    def sold_people(self) -> int:
        return sum(p.served_people + p.refund_people for p in self.plans.values())

    @property
    def moved_people(self) -> int:
        """服务日不等于原售日的人数（真正的改期）。"""
        total = 0
        for plan in self.plans.values():
            for bid, opt in plan.assignments.items():
                if opt is None:
                    continue
                b = self._booking(bid)
                if opt.day != b.first_day:
                    total += b.people
        return total

    @property
    def viewing_people(self) -> int:
        """最终按观礼安置的人数（含留在原售日、未改期的旅客）。"""
        return sum(p.viewing_people for p in self.plans.values())

    @property
    def tour_people(self) -> int:
        """最终按普通游线安置的人数。"""
        return sum(p.tour_people for p in self.plans.values())

    @property
    def viewing_rebooked_people(self) -> int:
        """改到非原售日观礼的人数。"""
        total = 0
        for plan in self.plans.values():
            for bid, opt in plan.assignments.items():
                if opt is None or opt.kind != "viewing":
                    continue
                b = self._booking(bid)
                if opt.day != b.first_day:
                    total += b.people
        return total

    @property
    def refund_people(self) -> int:
        return sum(p.refund_people for p in self.plans.values())

    @property
    def refund_cost(self) -> int:
        return sum(p.refund_cost for p in self.plans.values())

    @property
    def operator_cost(self) -> int:
        return sum(p.operator_cost for p in self.plans.values())

    @property
    def total_cost(self) -> int:
        return self.operator_cost + self.refund_cost + sum(p.passenger_cost for p in self.plans.values())

    @property
    def reserve_gap(self) -> int:
        """退费准备金按批设置：逐批核对，跨批缺口不能相互挪用。"""
        return sum(max(0, p.refund_cost - self.refund_cap) for p in self.plans.values())

    @property
    def event_refund_costs(self) -> dict[str, int]:
        return {e: p.refund_cost for e, p in self.plans.items()}

    @property
    def conflicts(self) -> list[Conflict]:
        return [c for p in self.plans.values() for c in p.conflicts]

    def _booking(self, bid: str) -> Booking:
        return self._dataset.bookings[bid]

    _dataset: Dataset = field(default=None, repr=False, compare=False)


def evaluate_term(
    dataset: Dataset,
    term: Terms,
    scenario: Scenario,
) -> ScenarioResult:
    plans: dict[str, Plan] = {}
    for event in dataset.events:
        first = min(b.first_day for b in dataset.bookings_by_event[event])
        if scenario.confirmed:
            actual = first + dt.timedelta(days=scenario.delay)
            known = actual - dt.timedelta(days=scenario.lead)
        else:
            # 窗口未定：仍以 scenario.delay 作为最终可能日，但此时不能锁定观礼
            actual = None
            known = first + dt.timedelta(days=scenario.delay - scenario.lead)
        # 换期选项只有在通告提前量满足条款最少通知日时才发得出
        lead_actual = (actual - known).days if actual is not None else scenario.lead
        exchange_available = lead_actual >= term.min_notice
        plans[event] = allocate(
            dataset, event, actual, term.services,
            deadline_days=term.deadline_days,
            exchange_available=exchange_available,
        )
    result = ScenarioResult(scenario, term, plans, term.max_refund)
    result._dataset = dataset
    return result


def grid(delays: Iterable[int] | None = None,
         leads: Iterable[int] | None = None) -> list[Scenario]:
    delays = list(delays if delays is not None else range(MAX_DELAY + 1))
    leads = list(leads if leads is not None else range(MAX_LEAD + 1))
    return [Scenario(d, l) for d in delays for l in leads]


def term_table(dataset: Dataset, scenarios: Sequence[Scenario]) -> list[dict]:
    rows = []
    for sc in scenarios:
        for term in dataset.terms:
            r = evaluate_term(dataset, term, sc)
            rows.append({
                "scenario": sc.name,
                "delay": sc.delay,
                "lead": sc.lead,
                "confirmed": sc.confirmed,
                "terms": term.name,
                "services": "+".join(term.services),
                "min_notice": term.min_notice,
                "deadline_days": term.deadline_days,
                "sold_people": r.sold_people,
                "moved_people": r.moved_people,
                "moved_share": r.moved_people / r.sold_people if r.sold_people else None,
                "viewing_people": r.viewing_people,
                "viewing_rebooked_people": r.viewing_rebooked_people,
                "tour_people": r.tour_people,
                "refund_people": r.refund_people,
                "refund_cost": r.refund_cost,
                "operator_cost": r.operator_cost,
                "total_cost": r.total_cost,
                "refund_cap": r.refund_cap,
                "reserve_gap": r.reserve_gap,
                "conflicts": len(r.conflicts),
                "infeasible": any(p.infeasible for p in r.plans.values()),
            })
    return rows


def shared_reservation(
    dataset: Dataset,
    term: Terms,
    delays: Iterable[int] = (0, 1, 2),
    lead: int = 0,
) -> dict:
    """互斥实际窗口下的共享预留，同时给出逐窗口简单相加的对照。

    同一资源键：跨互斥窗口取最大；同一窗口内各客群已经相加。
    """
    per_window = {}
    for d in delays:
        sc = Scenario(d, lead)
        res = evaluate_term(dataset, term, sc)
        usage: dict[tuple, int] = defaultdict(int)
        cohorts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for event, plan in res.plans.items():
            for bid, opt in plan.assignments.items():
                if opt is None:
                    continue
                b = dataset.bookings[bid]
                for key in resource_keys(opt.slot):
                    usage[key] += b.people
                    cohorts[key][str(b.cohort)] += b.people
        per_window[d] = {"usage": dict(usage), "cohorts": {k: dict(v) for k, v in cohorts.items()}}
    shared: dict[tuple, int] = {}
    naive: dict[tuple, int] = {}
    for d, info in per_window.items():
        for key, qty in info["usage"].items():
            shared[key] = max(shared.get(key, 0), qty)
            naive[key] = naive.get(key, 0) + qty
    return {
        "terms": term.name,
        "lead": lead,
        "per_window": per_window,
        "shared": shared,
        "naive_sum": naive,
        "overstated": {k: naive[k] - shared[k] for k in naive if naive[k] > shared[k]},
    }


@dataclass
class RiskBand:
    term: Terms
    delay: int
    lead: int
    low_refund_people: int
    high_refund_people: int
    low_refund_cost: int
    high_refund_cost: int
    low_total_cost: int
    high_total_cost: int
    reasons: list[str] = field(default_factory=list)
    reserve_gap_high: int = 0


def risk_band(
    dataset: Dataset,
    term: Terms,
    delay: int,
    lead: int,
    *,
    as_of: dt.date | None = None,
) -> RiskBand:
    """未确定窗口或旅客未应答时的风险边界。

    下界：窗口落在最有利的实际日、所有未决旅客都接受仍可安排的选项、容量足够。
    上界：窗口落在最不利的实际日、所有未决旅客按合同退费，并叠加退费准备金缺口。
    边界只使用 ``as_of`` 当日能知道的信息（默认取各批最终可见通告日），
    不把“可能被接受”当成已兑现，也不把 ``as_of`` 之后才收到的回复当成已知。
    """
    as_of = as_of or max(n.known for n in dataset.notices)
    reasons: list[str] = []
    lows: list[tuple[int, int, int, int]] = []
    highs: list[tuple[int, int, int, int]] = []

    for event in dataset.events:
        first = min(b.first_day for b in dataset.bookings_by_event[event])
        actual = first + dt.timedelta(days=delay)
        known = actual - dt.timedelta(days=lead)
        bookings = dataset.bookings_by_event[event]

        # 最好情形：全部可服务旅客都选最低成本的可接受选项
        exchange_available = lead >= term.min_notice
        best = allocate(
            dataset, event, actual, term.services,
            deadline_days=term.deadline_days,
            exchange_available=exchange_available,
        )
        low_people = best.refund_people
        low_cost = best.refund_cost
        low_total = best.total_cost

        # 最坏情形：截至 as_of 仍未应答的旅客按合同退费
        answered = {
            c.booking for c in dataset.choices
            if c.day <= as_of
        }
        unresolved = [b.id for b in bookings if b.id not in answered]
        if not unresolved:
            unresolved = [b.id for b in bookings if not b.existing]
        worst = allocate(
            dataset, event, actual, term.services,
            deadline_days=term.deadline_days,
            exchange_available=exchange_available,
            force_refund=unresolved,
        )
        high_people = worst.refund_people
        high_cost = worst.refund_cost
        high_total = worst.total_cost

        lows.append((low_people, low_cost, low_total, 0))
        # 准备金按批设置：每批单独核对缺口，不能把另一批的余额挪来补
        highs.append((high_people, high_cost, high_total, max(0, high_cost - term.max_refund)))

    low_people, low_cost, low_total, _ = map(sum, zip(*lows))
    high_people, high_cost, high_total, gap = map(sum, zip(*highs))
    reasons.append("退费准备金按批设置，上界缺口逐批相加，跨批不能相互挪用")
    reasons.append("上界假设未决旅客全部按合同退费，未把可能接受当作已兑现")
    reasons.append("下界假设窗口落在该延期日、全部可服务旅客都接受且容量足够")
    if not term.services or "观礼" not in term.services:
        reasons.append("条款不含观礼，观礼权利只能全部转入退费")
    return RiskBand(
        term=term, delay=delay, lead=lead,
        low_refund_people=low_people, high_refund_people=high_people,
        low_refund_cost=low_cost, high_refund_cost=high_cost,
        low_total_cost=low_total, high_total_cost=high_total,
        reasons=reasons, reserve_gap_high=gap,
    )


def open_window_risk(
    dataset: Dataset,
    term: Terms,
    possible_delays: Sequence[int],
    lead: int,
) -> dict:
    """窗口仍未唯一确定时的风险边界。

    只要没有确认通告，观礼席位不能锁定；有普通游线备选的订单可先按约定锁定。
    对每个可能日分别求最好/最坏，再取边界；不假设会落在对自己最有利的那一天。
    """
    bands = [risk_band(dataset, term, d, lead) for d in possible_delays]
    return {
        "terms": term.name,
        "possible_delays": list(possible_delays),
        "lead": lead,
        "refund_cost_low": min(b.low_refund_cost for b in bands),
        "refund_cost_high": max(b.high_refund_cost for b in bands),
        "refund_people_low": min(b.low_refund_people for b in bands),
        "refund_people_high": max(b.high_refund_people for b in bands),
        "total_cost_low": min(b.low_total_cost for b in bands),
        "total_cost_high": max(b.high_total_cost for b in bands),
        "reserve_gap_high": max(b.reserve_gap_high for b in bands),
        "note": "窗口未定时观礼不能锁定；不能把某一个可能日当作已确认",
    }


def behavior_estimate(
    dataset: Dataset,
    as_of: dt.date,
    *,
    tour: str | None = None,
    event: str | None = None,
) -> list[dict]:
    """按销售组/优先层给出可改期比例与区间，明确样本量。"""
    stats = cohort_stats(dataset, as_of, event=event)
    rows = []
    for key, st in sorted(stats.items(), key=lambda kv: kv[0].label):
        if tour is not None and key.tour != tour:
            continue
        rate = st.reschedule_rate
        ci = rate.wilson()
        rows.append({
            "cohort": key.label,
            "tour": key.tour,
            "priority": key.priority,
            "bookings": st.bookings,
            "people": st.people,
            "responded": st.responded,
            "reschedule_n": st.outcomes.get("改到观礼", 0),
            "reschedule_rate": rate.point,
            "reschedule_ci_low": ci[0] if ci else None,
            "reschedule_ci_high": ci[1] if ci else None,
            "tour_n": st.outcomes.get("同意普通游线", 0),
            "refund_n": st.outcomes.get("明确退费", 0),
            "no_reply_n": st.outcomes.get("无应答", 0),
            "evidence": "有样本" if st.responded else "无回复样本",
        })
    return rows


def cost_distribution(result: ScenarioResult) -> dict[str, dict]:
    """逐订单成本分布（改期、换普通游线、退费），按人计。

    换普通游线的“成本”是改期后车组与点位的合用量成本，会随当日剩余容量变化；
    退款按合同人均退费。分布保留最小/中位/最大，避免只报一个平均数。
    """
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)  # (unit, people)
    for plan in result.plans.values():
        for bid, opt in plan.assignments.items():
            b = dataset_booking(result, bid)
            if opt is None:
                cat, unit = "退款", b.refund
            elif opt.kind == "viewing":
                cat = "改到观礼" if opt.day != b.first_day else "留在原售日"
                unit = 0
            else:
                cat, unit = "换普通游线", opt.slot.car_cost + opt.slot.site_cost
            buckets[cat].append((unit, b.people))
    out = {}
    for cat, pairs in buckets.items():
        units = sorted(u for u, _ in pairs)
        n = len(units)
        out[cat] = {
            "bookings": n,
            "people": sum(p for _, p in pairs),
            "per_person_min": units[0],
            "per_person_median": units[n // 2],
            "per_person_max": units[-1],
            "cost_total": sum(u * p for u, p in pairs),
        }
    return out


def dataset_booking(result: ScenarioResult, bid: str) -> Booking:
    return result._dataset.bookings[bid]


def expected_behavior_shift(
    dataset: Dataset,
    as_of: dt.date,
    term: Terms,
    scenario: Scenario,
) -> dict:
    """把历史分层回复率套到当前情景，给出“可改期比例”的估计区间。

    只在同一销售组内按层应用；无回复样本的层不借点估计，改为在区间两端分别取
    0 和 1（即该层全部不改期 / 全部改期）。因此区间来自“证据缺口”，不是随意加减。
    """
    stats = cohort_stats(dataset, as_of)
    rows = []
    for event in dataset.events:
        # 未决窗只表示“此刻仍看不出来”，不把未决当成可改期权利
        st = dataset.state(event, as_of)
        for b in dataset.bookings_by_event[event]:
            if st.confirmed is None:
                rows.append({
                    "booking": b.id, "event": event, "people": b.people,
                    "改到观礼": 0.0, "同意普通游线": 0.0, "明确退费": 0.0,
                    "无应答": 1.0, "no_evidence": True, "window_open": True,
                })
                continue
            p = outcome_probabilities(stats, b)
            rows.append({"booking": b.id, "event": event, "people": b.people,
                         "window_open": False, **p})
    people = sum(r["people"] for r in rows)

    def share(key: str, *, no_evidence_value: float) -> float | None:
        total = 0.0
        for r in rows:
            if r.get("no_evidence"):
                total += r["people"] * no_evidence_value
            else:
                total += r["people"] * r[key]
        return total / people if people else None

    return {
        "terms": term.name,
        "scenario": scenario.name,
        "as_of": as_of.isoformat(),
        "people": people,
        "reschedule_share_low": share("改到观礼", no_evidence_value=0.0),
        "reschedule_share_high": share("改到观礼", no_evidence_value=1.0),
        "tour_share_point": share("同意普通游线", no_evidence_value=0.0),
        "refund_share_point": share("明确退费", no_evidence_value=0.0),
        "no_reply_share_point": share("无应答", no_evidence_value=1.0),
        "no_evidence_people": sum(r["people"] for r in rows if r.get("no_evidence")),
        "open_window_people": sum(r["people"] for r in rows if r.get("window_open")),
        "note": "按销售组分层；无样本层在区间两端分别取 0 和 1；窗口未定时不计入改期",
    }
