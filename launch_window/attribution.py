"""事后成本与签条款时的风险放在一起回看。

把实际观察到的补偿分成两部分：
* 窗口变化造成：即使容量充足、所有可接受选择都还在，也必须改期或退费的部分。
* 选择被提前发完造成：因为前面旅客按优先次序占满容量，后面的旅客只能改选或退费。

划分方式是对同一实际窗口做两次求解：
1. ``unconstrained``：忽略可用量，只看窗口本身是否让订单无法留在原计划；
2. ``capacity``：按当时的优先次序与真实容量求解。
difference = capacity - unconstrained 即为容量竞争造成的增量。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

from .allocate import allocate
from .model import Dataset


@dataclass
class Attribution:
    event: str
    actual: dt.date | None
    observed_people: int
    observed_cost: int
    window_refund_people: int
    window_refund_cost: int
    capacity_refund_people: int
    capacity_refund_cost: int
    promised_viewing_people: int = 0
    promised_viewing_mismatch_people: int = 0
    promised_tour_people: int = 0
    promised_unacceptable_people: int = 0

    @property
    def capacity_extra_people(self) -> int:
        return max(0, self.capacity_refund_people - self.window_refund_people)

    @property
    def capacity_extra_cost(self) -> int:
        return max(0, self.capacity_refund_cost - self.window_refund_cost)


def observed_costs(dataset: Dataset, event: str) -> tuple[int, int]:
    people = cost = 0
    for c in dataset.choices:
        b = dataset.bookings.get(c.booking)
        if b is None or b.event != event:
            continue
        people += b.people
        cost += b.people * c.actual_per_unit
    return people, cost


def promise_counts(dataset: Dataset, event: str) -> dict[str, int]:
    """已结算承诺里无法按承诺兑现的部分。

    这些是"承诺被提前发出"的直接证据：承诺的观礼日不是最终确认日、或接受了
    订单本不可接受的普通游线。它们不并入总体退款率，而是单独计数。
    """
    actual = dataset.confirmed_day(event)
    out = {
        "promised_viewing_people": 0,
        "promised_viewing_mismatch_people": 0,
        "promised_tour_people": 0,
        "promised_unacceptable_people": 0,
    }
    for c in dataset.choices:
        b = dataset.bookings.get(c.booking)
        if b is None or b.event != event:
            continue
        if c.choice == "改到观礼":
            out["promised_viewing_people"] += b.people
            if actual is not None and c.offer_day != actual:
                out["promised_viewing_mismatch_people"] += b.people
        elif c.choice == "同意普通游线":
            out["promised_tour_people"] += b.people
            if c.offer_day not in b.allowed or "游线晚" not in b.fallback:
                out["promised_unacceptable_people"] += b.people
    return out


def attribute_event(
    dataset: Dataset,
    event: str,
    services: Iterable[str] = ("观礼", "游线早", "游线晚"),
    *,
    deadline_days: int | None = None,
) -> Attribution:
    actual = dataset.confirmed_day(event)
    observed_people, observed_cost = observed_costs(dataset, event)
    promises = promise_counts(dataset, event)
    if actual is None:
        return Attribution(
            event, None, observed_people, observed_cost,
            0, 0, 0, 0, **promises,
        )
    window = allocate(
        dataset, event, actual, services, deadline_days=deadline_days,
        ignore_capacity=True,
    )
    actual_plan = allocate(
        dataset, event, actual, services, deadline_days=deadline_days,
    )
    return Attribution(
        event=event,
        actual=actual,
        observed_people=observed_people,
        observed_cost=observed_cost,
        window_refund_people=window.refund_people,
        window_refund_cost=window.refund_cost,
        capacity_refund_people=actual_plan.refund_people,
        capacity_refund_cost=actual_plan.refund_cost,
        **promises,
    )


def attribute(dataset: Dataset, **kwargs) -> list[Attribution]:
    return [attribute_event(dataset, e, **kwargs) for e in dataset.events]


def summary(rows: Iterable[Attribution]) -> dict[str, int]:
    rows = list(rows)
    return {
        "observed_people": sum(r.observed_people for r in rows),
        "observed_cost": sum(r.observed_cost for r in rows),
        "window_refund_people": sum(r.window_refund_people for r in rows),
        "window_refund_cost": sum(r.window_refund_cost for r in rows),
        "capacity_refund_people": sum(r.capacity_refund_people for r in rows),
        "capacity_refund_cost": sum(r.capacity_refund_cost for r in rows),
        "capacity_extra_people": sum(r.capacity_extra_people for r in rows),
        "capacity_extra_cost": sum(r.capacity_extra_cost for r in rows),
        "promised_viewing_people": sum(r.promised_viewing_people for r in rows),
        "promised_viewing_mismatch_people": sum(
            r.promised_viewing_mismatch_people for r in rows),
        "promised_tour_people": sum(r.promised_tour_people for r in rows),
        "promised_unacceptable_people": sum(
            r.promised_unacceptable_people for r in rows),
    }
