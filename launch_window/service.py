"""客服在通告当日登记条件、查询可告知选项、记录发出与接受结果。

本模块只回答“此刻能说哪些”，不替旅客预选，也不把尚有效的承诺当成已释放。
可告知选项按以下顺序收窄：
1. 只使用通告日已经发出的信息（观礼日未唯一确定前不报观礼）；
2. 订单可接受日期与可接受产品；
3. 订单自身的回复截止与新条款的最晚接受日；
4. 隔离已接受（锁定）与已发未答（占用）的容量，剩余容量按可同时接受核对。
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

from .allocate import Option, acceptance_cutoff, options_for, resource_keys
from .ledger import ACCEPTED, HOLD, build_ledger
from .model import Booking, Dataset, NoticeState, Terms


@dataclass
class Tellable:
    booking: Booking
    options: list[Option]
    reason: str
    blocked: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.options]


@dataclass
class NoticeRegistration:
    event: str
    as_of: dt.date
    state: NoticeState
    terms: Terms
    tellable: dict[str, Tellable]
    sent: list[dict]
    accepted: list[dict]
    held: list[dict]
    conflicts: list[str]

    def rows(self) -> list[dict]:
        return [
            {
                "event": self.event,
                "as_of": self.as_of.isoformat(),
                "notice_seq": self.state.seq,
                "kind": self.state.kind,
                "possible_days": [d.isoformat() for d in self.state.possible],
                "confirmed": self.state.confirmed.isoformat() if self.state.confirmed else None,
                "terms": self.terms.name,
            }
        ]


def _capacity_snapshot(
    dataset: Dataset, event: str,
) -> dict[tuple, int]:
    caps: dict[tuple, int] = {}
    for s in dataset.slots:
        if s.event != event:
            continue
        car, site = resource_keys(s)
        caps[car] = s.car_seats
        caps[site] = s.site_seats
    return caps


def _existing_occupancy(
    dataset: Dataset, event: str, as_of: dt.date,
) -> tuple[dict[tuple, int], set[str], list[dict]]:
    """已接受与已发未答占用的容量，以及不再接受新查询的订单。"""
    used: dict[tuple, int] = defaultdict(int)
    blocked_bookings: set[str] = set()
    held_rows: list[dict] = []
    for e in build_ledger(dataset, event, as_of):
        if e.slot is None:
            continue
        keys = resource_keys(e.slot)
        if e.status == "locked":
            for k in keys:
                used[k] += e.booking.people
            # 已锁定一条替代，后续不再给同一订单发相斥选项
            blocked_bookings.add(e.booking.id)
        elif e.status == "held":
            for k in keys:
                used[k] += e.booking.people
            blocked_bookings.add(e.booking.id)
        if e.status in ("locked", "held"):
            held_rows.append({
                "offer": e.offer.id,
                "booking": e.booking.id,
                "product": e.offer.product,
                "day": e.offer.option_day.isoformat(),
                "status": e.status,
                "people": e.booking.people,
            })
    return used, blocked_bookings, held_rows


def query_tellable(dataset: Dataset, event: str, as_of: dt.date, terms: Terms) -> dict[str, Tellable]:
    """在通告当日查询每个订单可被明确告知的选项及占用后剩余容量。"""
    st = dataset.state(event, as_of)
    caps = _capacity_snapshot(dataset, event)
    used, blocked_bookings, _ = _existing_occupancy(dataset, event, as_of)
    residual = {k: caps[k] - used.get(k, 0) for k in caps}

    out: dict[str, Tellable] = {}
    for b in dataset.bookings_by_event[event]:
        if b.id in blocked_bookings:
            out[b.id] = Tellable(b, [], "已锁定或已发未答，暂不重复发选项")
            continue
        cutoff = acceptance_cutoff(b, terms.deadline_days)
        opts = options_for(
            dataset, b, st.confirmed, terms.services,
            cutoff=cutoff,
            exchange_available=terms.min_notice <= (st.confirmed - st.known).days
            if st.confirmed else False,
        )
        # 仍有效的相互竞争订单占用按可同时接受的数量核对：逐项检查剩余容量
        keep: list[Option] = []
        blocked: list[str] = []
        for opt in opts:
            keys = resource_keys(opt.slot)
            slack = min(residual.get(k, 0) for k in keys)
            if slack >= b.people:
                keep.append(opt)
            else:
                tight = min(keys, key=lambda k: residual.get(k, 0))
                blocked.append(
                    f"{opt.label} 需 {b.people} 人，{tight[0]} {tight[2].isoformat()} "
                    f"{tight[3]} 余 {residual.get(tight, 0)}/{caps[tight]}"
                )
        if not keep:
            reason = "无可告知选项" if not opts else "可接受选项均被占用"
            out[b.id] = Tellable(b, [], reason, blocked)
        else:
            out[b.id] = Tellable(b, keep, "可告知", blocked)
    return out


def register_notice(
    dataset: Dataset, event: str, as_of: dt.date, terms: Terms,
) -> NoticeRegistration:
    """登记某条通告当日的条件、可告知选项与实际发出/接受结果。"""
    st = dataset.state(event, as_of)
    tellable = query_tellable(dataset, event, as_of, terms)
    sent, accepted, held = [], [], []
    for o in dataset.offers_by_event.get(event, []):
        if o.sent > as_of:
            continue
        row = {
            "offer": o.id,
            "booking": o.booking,
            "product": o.product,
            "day": o.option_day.isoformat(),
            "state": o.state,
            "sent": o.sent.isoformat(),
            "people": dataset.bookings[o.booking].people,
        }
        sent.append(row)
        if o.state == ACCEPTED:
            accepted.append(row)
        elif o.state == HOLD:
            held.append(row)
    conflicts: list[str] = []
    for o in dataset.offers_by_event.get(event, []):
        if o.sent > as_of:
            continue
        b = dataset.bookings[o.booking]
        if o.product == "观礼":
            if st.confirmed is None:
                conflicts.append(f"{o.id} 在观礼日未唯一确定时发出观礼选项")
            elif st.confirmed != o.option_day:
                conflicts.append(f"{o.id} 观礼日与已确认日不一致")
        if o.product != "观礼" and o.product not in b.fallback:
            conflicts.append(f"{o.id} {o.product} 不在订单可接受产品")
        if o.option_day not in b.allowed:
            conflicts.append(f"{o.id} {o.option_day} 不在订单可接受日期")
    return NoticeRegistration(
        event=event, as_of=as_of, state=st, terms=terms,
        tellable=tellable, sent=sent, accepted=accepted, held=held,
        conflicts=conflicts,
    )


def registrations(dataset: Dataset, terms: Terms) -> list[NoticeRegistration]:
    """对每条实际发出的通告逐一登记。"""
    out = []
    for event in dataset.events:
        for n in dataset.notices_by_event[event]:
            out.append(register_notice(dataset, event, n.known, terms))
    return out


@dataclass
class OfferImpact:
    booking: str
    option: Option
    affected: dict[str, int]  # booking -> 该订单因此少掉的选项数
    lost_all: list[str]      # 因此完全没有可告知选项、只能等退费的订单
    blocked: dict[str, list[str]]


def _residual(
    dataset: Dataset, event: str, as_of: dt.date, terms: Terms,
    extra: list[tuple[str, Option]] = (),
) -> tuple[dict[tuple, int], dict[tuple, int]]:
    caps = _capacity_snapshot(dataset, event)
    used, blocked, _ = _existing_occupancy(dataset, event, as_of)
    used = dict(used)
    for bid, opt in extra:
        for k in resource_keys(opt.slot):
            used[k] = used.get(k, 0) + dataset.bookings[bid].people
    return {k: caps[k] - used.get(k, 0) for k in caps}, caps


def _tellable_for(
    dataset: Dataset, event: str, as_of: dt.date, terms: Terms, b: Booking,
    residual: dict[tuple, int], caps: dict[tuple, int],
) -> tuple[list[Option], list[str]]:
    st = dataset.state(event, as_of)
    opts = options_for(
        dataset, b, st.confirmed, terms.services,
        cutoff=acceptance_cutoff(b, terms.deadline_days),
        exchange_available=terms.min_notice <= (st.confirmed - st.known).days
        if st.confirmed else False,
    )
    keep, blocked = [], []
    for opt in opts:
        keys = resource_keys(opt.slot)
        slack = min(residual.get(k, 0) for k in keys)
        if slack >= b.people:
            keep.append(opt)
        else:
            tight = min(keys, key=lambda k: residual.get(k, 0))
            blocked.append(
                f"{opt.label} 需 {b.people} 人，{tight[0]} {tight[2].isoformat()} "
                f"{tight[3]} 余 {residual.get(tight, 0)}/{caps[tight]}"
            )
    return keep, blocked


def impact_of_offer(
    dataset: Dataset, event: str, as_of: dt.date, terms: Terms,
    booking_id: str, option: Option,
) -> OfferImpact:
    """如果把某条备选说给某位客人，其他客人能选的范围会缩小多少。

    这里只比较“要不要发这条”的边际影响，不把查询本身当成已发出的承诺。
    """
    residual, caps = _residual(dataset, event, as_of, terms)
    affected: dict[str, int] = {}
    lost_all: list[str] = []
    blocked: dict[str, list[str]] = {}
    after_residual = dict(residual)
    for k in resource_keys(option.slot):
        after_residual[k] = residual.get(k, 0) - dataset.bookings[booking_id].people
    for b in dataset.bookings_by_event[event]:
        if b.id == booking_id:
            continue
        before, _ = _tellable_for(dataset, event, as_of, terms, b, residual, caps)
        after, why = _tellable_for(dataset, event, as_of, terms, b, after_residual, caps)
        if len(after) < len(before):
            affected[b.id] = len(before) - len(after)
            if before and not after:
                lost_all.append(b.id)
            if why:
                blocked[b.id] = why
    return OfferImpact(booking_id, option, affected, lost_all, blocked)


@dataclass
class CommitmentRun:
    order: list[str]
    assigned: dict[str, Option]
    no_option: list[str]
    refused_commit: list[tuple[str, str]]


def simulate_commitments(
    dataset: Dataset, event: str, as_of: dt.date, terms: Terms,
    *, prefer: str = "first",
) -> CommitmentRun:
    """按优先次序依次给出可告知选项，观察前一位客人如何缩小后一位的范围。

    ``prefer`` 只决定当面先报哪一项（按标签首个或成本最低），不代表客人接受；
    这里模拟的是“承诺已发出并占用”，不是把预计接受当成已兑现。
    """
    bookings = sorted(
        dataset.bookings_by_event[event],
        key=lambda b: (not b.existing, b.priority, b.id),
    )
    residual, caps = _residual(dataset, event, as_of, terms)
    _, blocked_bookings, _ = _existing_occupancy(dataset, event, as_of)
    taken: list[tuple[str, Option]] = []
    assigned: dict[str, Option] = {}
    no_option: list[str] = []
    refused: list[tuple[str, str]] = []
    for b in bookings:
        if b.id in blocked_bookings:
            # 已经发出过选项且尚未释放，不重复占位，也不计为“只能退费”
            assigned[b.id] = None
            continue
        tell, why = _tellable_for(dataset, event, as_of, terms, b, residual, caps)
        if not tell:
            no_option.append(b.id)
            if why:
                refused.append((b.id, "；".join(why)))
            continue
        if prefer == "cheapest":
            opt = min(tell, key=lambda o: o.slot.operator_cost)
        else:
            opt = tell[0]
        assigned[b.id] = opt
        taken.append((b.id, opt))
        for k in resource_keys(opt.slot):
            residual[k] = residual.get(k, 0) - b.people
    return CommitmentRun(
        order=[b.id for b in bookings],
        assigned=assigned,
        no_option=no_option,
        refused_commit=refused,
    )
