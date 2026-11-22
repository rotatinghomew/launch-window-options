"""按“当时能知道的信息”重演，并按销售组分层估计选择分布。

原则
----
* 只用 ``known <= as_of`` 的通告；``as_of`` 之后才发出的通告一律不可见。
* 只用 ``day <= as_of`` 的旅客回复；此后的回复不进入当时的估计。
* 估计按销售组（``tour``）与签约时已知的优先/保留方式/可接受产品分层，
  不把普通散客或其它销售组的历史改期率直接套到观礼团。
* 无样本的层明确标为“无依据”，不借用其它层的点估计。
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .model import Booking, Dataset, NoticeState

RESCHEDULE = "改到观礼"
TOUR = "同意普通游线"
REFUND = "明确退费"
NO_REPLY = "无应答"
OUTCOMES = (RESCHEDULE, TOUR, REFUND, NO_REPLY)


@dataclass(frozen=True)
class CohortKey:
    tour: str
    priority: int
    option: str
    fallback: tuple[str, ...]

    @classmethod
    def of(cls, b: Booking) -> "CohortKey":
        return cls(b.tour, b.priority, b.option, b.fallback)

    @property
    def label(self) -> str:
        fb = "+".join(self.fallback) if self.fallback else "无"
        return f"{self.tour}/优先{self.priority}/{self.option}/备选{fb}"


@dataclass
class Rate:
    successes: int
    total: int

    @property
    def point(self) -> float | None:
        return None if self.total == 0 else self.successes / self.total

    def wilson(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson 区间；小样本时比点估计更适合直接展示。"""
        n = self.total
        if n == 0:
            return None
        p = self.successes / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class CohortStats:
    key: CohortKey
    bookings: int = 0
    people: int = 0
    outcomes: Counter = field(default_factory=Counter)
    people_by_outcome: Counter = field(default_factory=Counter)

    @property
    def responded(self) -> int:
        return sum(v for k, v in self.outcomes.items() if k != NO_REPLY)

    @property
    def response_rate(self) -> Rate:
        return Rate(self.responded, self.bookings)

    def outcome_share(self, outcome: str) -> Rate:
        return Rate(self.outcomes.get(outcome, 0), self.responded)

    @property
    def reschedule_rate(self) -> Rate:
        return Rate(self.outcomes.get(RESCHEDULE, 0), self.bookings)


@dataclass
class EventReplay:
    event: str
    as_of: dt.date
    state: NoticeState
    bookings: list[Booking]
    observed: dict[str, str]
    stats: list[CohortStats]
    unresolved: list[Booking]
    answered_people: int
    total_people: int

    @property
    def reschedule_people(self) -> int:
        return sum(
            b.people for b in self.bookings
            if self.observed.get(b.id) == RESCHEDULE
        )

    @property
    def refund_people(self) -> int:
        return sum(
            b.people for b in self.bookings
            if self.observed.get(b.id) == REFUND
        )


def observed_choices(dataset: Dataset, as_of: dt.date) -> dict[str, str]:
    return {
        c.booking: c.choice
        for c in dataset.choices
        if c.day <= as_of
    }


def cohort_stats(
    dataset: Dataset,
    as_of: dt.date,
    *,
    event: str | None = None,
    include_events: Iterable[str] | None = None,
) -> dict[CohortKey, CohortStats]:
    """截至 as_of 的分层选择统计。"""
    observed = observed_choices(dataset, as_of)
    allow = set(include_events) if include_events is not None else None
    out: dict[CohortKey, CohortStats] = {}
    for b in dataset.parties:
        if event is not None and b.event != event:
            continue
        if allow is not None and b.event not in allow:
            continue
        key = CohortKey.of(b)
        st = out.setdefault(key, CohortStats(key))
        st.bookings += 1
        st.people += b.people
        outcome = observed.get(b.id)
        if outcome is None:
            outcome = NO_REPLY
        st.outcomes[outcome] += 1
        st.people_by_outcome[outcome] += b.people
    return out


def replay(dataset: Dataset, as_of: dt.date) -> dict[str, EventReplay]:
    """重演 as_of 当日各批次的可见状态、回复与未决量。"""
    observed = observed_choices(dataset, as_of)
    out: dict[str, EventReplay] = {}
    for event in dataset.events:
        st = dataset.state(event, as_of)
        bookings = dataset.bookings_by_event[event]
        stats = list(cohort_stats(
            dataset, as_of, event=event,
        ).values())
        unresolved = [b for b in bookings if b.id not in observed]
        out[event] = EventReplay(
            event=event,
            as_of=as_of,
            state=st,
            bookings=bookings,
            observed={b.id: observed[b.id] for b in bookings if b.id in observed},
            stats=stats,
            unresolved=unresolved,
            answered_people=sum(b.people for b in bookings if b.id in observed),
            total_people=sum(b.people for b in bookings),
        )
    return out


def timeline(dataset: Dataset) -> list[EventReplay]:
    """逐条通告推演；每条通告只看当日及之前已经知道的信息。"""
    seen: set[tuple[str, int]] = set()
    out: list[EventReplay] = []
    moments = sorted({n.known for n in dataset.notices})
    for as_of in moments:
        for event in dataset.events:
            st = dataset.state(event, as_of)
            if st.seq == 0 or (event, st.seq) in seen:
                continue
            seen.add((event, st.seq))
            out.append(replay(dataset, as_of)[event])
    return out


def outcome_probabilities(
    stats: Mapping[CohortKey, CohortStats],
    booking: Booking,
    *,
    smoothing: float = 0.5,
    allow_pooled_fallback: bool = False,
) -> dict[str, float]:
    """订单的回复/结果概率。

    只用同一销售组的同一层；该层没有回复样本时返回均匀分布并标记 ``no_evidence``，
    默认不借用其它层。``allow_pooled_fallback`` 仅用于做敏感性对照，不作为默认依据。
    """
    key = CohortKey.of(booking)
    st = stats.get(key)
    if st is None or st.responded == 0:
        if allow_pooled_fallback:
            # 只借同一销售组内其它优先层的回复构成，绝不跨销售组
            same_tour = [s for k, s in stats.items() if k.tour == key.tour and s.responded]
            if same_tour:
                return _smoothed_outcomes(same_tour, smoothing) | {"no_evidence": 1.0}
        return {
            RESCHEDULE: 0.25, TOUR: 0.25, REFUND: 0.25, NO_REPLY: 0.25,
            "no_evidence": 1.0,
        }
    p = _smoothed_outcomes([st], smoothing)
    p["no_evidence"] = 0.0
    return p


def _smoothed_outcomes(stats: Sequence[CohortStats], smoothing: float) -> dict[str, float]:
    counts = Counter()
    for st in stats:
        counts.update(st.outcomes)
    for k in OUTCOMES:
        counts.setdefault(k, 0)
    n = sum(counts.values())
    k = len(OUTCOMES)
    denom = n + smoothing * k
    return {o: (counts[o] + smoothing) / denom for o in OUTCOMES}


def weighted_rate(
    stats: Mapping[CohortKey, CohortStats],
    outcome: str,
) -> Rate:
    return Rate(
        sum(s.outcomes.get(outcome, 0) for s in stats.values()),
        sum(s.bookings for s in stats.values()),
    )


def per_event_table(replays: Mapping[str, EventReplay]) -> list[dict]:
    rows = []
    for event, rp in replays.items():
        delay = None
        if rp.state.confirmed is not None:
            delay = (rp.state.confirmed - dataset_initial(rp)).days
        rows.append({
            "event": event,
            "seq": rp.state.seq,
            "kind": rp.state.kind,
            "known": rp.state.known.isoformat(),
            "possible_days": [d.isoformat() for d in rp.state.possible],
            "confirmed": rp.state.confirmed.isoformat() if rp.state.confirmed else None,
            "delay": delay,
            "bookings": len(rp.bookings),
            "people": rp.total_people,
            "answered_people": rp.answered_people,
            "unresolved_people": rp.total_people - rp.answered_people,
            "reschedule_people": rp.reschedule_people,
            "refund_people": rp.refund_people,
        })
    return rows


def dataset_initial(rp: EventReplay) -> dt.date:
    return min(b.first_day for b in rp.bookings)
