"""已发出承诺的台账与执行。

规则
----
* ``已接受``：锁定，新分配必须优先兑现。
* ``已发未答``：在订单回复截止日之前继续占用容量；即使新的试算想要这些座位，
  也不能替客服放掉。
* ``已拒绝``：释放。
* 同一订单已锁定一条替代后，其余仍被占用的选择才按合同释放。
* 回复时限已到且确实没有新通告时，本模块默认仍不擅自释放；只有调用方显式声明
  执行合同释放（``release_expired=True``）才释放，并逐条留下记录。
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .allocate import Conflict, Option, Plan, allocate, options_for, resource_keys
from .model import Booking, Dataset, Offer, Slot

HOLD = "已发未答"
ACCEPTED = "已接受"
REFUSED = "已拒绝"


@dataclass
class LedgerEntry:
    offer: Offer
    booking: Booking
    slot: Slot | None
    status: str  # "locked" | "held" | "released" | "invalid"
    reason: str
    release_condition: str | None = None


@dataclass
class Execution:
    event: str
    as_of: dt.date
    entries: list[LedgerEntry]
    released: list[LedgerEntry]
    conflicts: list[Conflict]
    plan: Plan | None = None

    def by_status(self, status: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.status == status]


def _slot_for(dataset: Dataset, booking: Booking, offer: Offer) -> Slot | None:
    return dataset.slot(booking.event, offer.option_day, offer.product)


def build_ledger(
    dataset: Dataset,
    event: str,
    as_of: dt.date,
    *,
    release_expired: bool = False,
) -> list[LedgerEntry]:
    """按 as_of 当日状态逐条判定承诺。"""
    entries: list[LedgerEntry] = []
    per_booking: dict[str, list[Offer]] = defaultdict(list)
    for o in dataset.offers_by_event.get(event, []):
        per_booking[o.booking].append(o)

    for bid, offers in per_booking.items():
        b = dataset.bookings[bid]
        locked = [o for o in offers if o.state == ACCEPTED]
        # 已锁定一条替代：其余未答选项按合同释放
        locked_product = locked[0] if locked else None
        for o in sorted(offers, key=lambda x: x.sent):
            slot = _slot_for(dataset, b, o)
            if o.state == REFUSED:
                entries.append(LedgerEntry(
                    o, b, slot, "released", "旅客已明确拒绝",
                    "旅客明确拒绝",
                ))
                continue
            if o.state == ACCEPTED:
                valid = _offer_valid(dataset, b, o, as_of)
                entries.append(LedgerEntry(
                    o, b, slot, "locked" if valid[0] else "invalid",
                    "已接受，优先兑现" if valid[0] else valid[1],
                ))
                continue
            # 已发未答
            if locked_product is not None:
                entries.append(LedgerEntry(
                    o, b, slot, "released",
                    f"同一订单已锁定 {locked_product.id}，其余选择按合同释放",
                    "已锁定一条替代",
                ))
                continue
            if o.sent > b.until:
                entries.append(LedgerEntry(
                    o, b, slot, "released", "发出日晚于订单回复截止",
                    "合同期限已过",
                ))
                continue
            if as_of > b.until:
                if release_expired:
                    entries.append(LedgerEntry(
                        o, b, slot, "released",
                        f"回复截止 {b.until.isoformat()} 已过且无新通告，按合同释放",
                        "回复时限已到",
                    ))
                else:
                    entries.append(LedgerEntry(
                        o, b, slot, "held",
                        f"回复截止 {b.until.isoformat()} 已过，但未获准执行释放，继续占用",
                    ))
                continue
            entries.append(LedgerEntry(
                o, b, slot, "held",
                f"已发未答，保留至 {b.until.isoformat()}",
            ))
    return entries


def _offer_valid(
    dataset: Dataset, b: Booking, offer: Offer, as_of: dt.date,
) -> tuple[bool, str]:
    if offer.option_day not in b.allowed:
        return False, f"日期 {offer.option_day.isoformat()} 不在订单可接受日期"
    if offer.product != "观礼" and offer.product not in b.fallback:
        return False, f"产品 {offer.product} 不在订单可接受产品"
    if offer.product == "观礼":
        st = dataset.state(b.event, max(offer.sent, as_of))
        if st.confirmed is None:
            return False, "观礼日尚未由正式通告唯一确定，暂不能锁定观礼席位"
        if st.confirmed != offer.option_day:
            return False, f"已确认观礼日为 {st.confirmed.isoformat()}，与告知日不符"
    return True, "有效"


def execute(
    dataset: Dataset,
    event: str,
    term_services: Iterable[str],
    *,
    as_of: dt.date,
    actual: dt.date | None = None,
    deadline_days: int | None = None,
    release_expired: bool = False,
    extra_locked: Mapping[str, Option] | None = None,
) -> Execution:
    """把已发出的承诺放进同一容量模型执行。

    已接受且有效的选项锁定；已发未答占用容量；无效的已接受选项不再静默兑现，
    而是登记具体冲突并保留仍可给的替代。
    """
    services = tuple(term_services)
    entries = build_ledger(dataset, event, as_of, release_expired=release_expired)
    locked: dict[str, Option] = dict(extra_locked or {})
    reserved: list[tuple[str, Option, int]] = []
    conflicts: list[Conflict] = []
    for e in entries:
        if e.slot is None:
            conflicts.append(Conflict(
                e.booking.id, "offer_slot_missing",
                f"{e.offer.id} 指向的 {e.offer.option_day} {e.offer.product} 无运力记录",
            ))
            continue
        opt = _option_from(dataset, e)
        if opt is None:
            continue
        if e.status == "locked":
            locked[e.booking.id] = opt
        elif e.status == "held":
            reserved.append((e.booking.id, opt, e.booking.people))
        elif e.status == "invalid":
            alternatives = _retainable(dataset, e.booking, actual, services, deadline_days)
            conflicts.append(Conflict(
                e.booking.id, "accepted_promise_conflict",
                f"{e.offer.id} 已接受但无法兑现：{e.reason}；仍可保留的替代："
                + ("；".join(alternatives) if alternatives else "无，需退费"),
            ))

    plan = allocate(
        dataset, event, actual, services,
        deadline_days=deadline_days,
        locked=locked,
        reserved=reserved,
    )
    # 已发出但无法兑现的承诺：逐条说明冲突和仍可保留的替代，不用总体比例抹平
    for e in entries:
        if e.status != "invalid":
            continue
        if any(c.booking == e.booking.id and c.code == "accepted_promise_conflict"
               for c in conflicts):
            continue
        alternatives = _retainable(dataset, e.booking, actual, services, deadline_days)
        plan.conflicts.append(Conflict(
            e.booking.id, "accepted_promise_conflict",
            f"{e.offer.id} 已接受但无法兑现：{e.reason}；仍可保留的替代："
            + ("；".join(alternatives) if alternatives else "无，需退费"),
        ))
    plan.conflicts.extend(conflicts)
    return Execution(
        event=event, as_of=as_of, entries=entries,
        released=[e for e in entries if e.status == "released"],
        conflicts=conflicts, plan=plan,
    )


def _option_from(dataset: Dataset, e: LedgerEntry) -> Option | None:
    slot = e.slot
    if slot is None:
        return None
    kind = "viewing" if e.offer.product == "观礼" else "tour"
    return Option(e.offer.option_day, e.offer.product, kind, slot)


def _retainable(
    dataset: Dataset,
    booking: Booking,
    actual: dt.date | None,
    services: Iterable[str],
    deadline_days: int | None,
) -> list[str]:
    from .allocate import acceptance_cutoff

    if actual is None:
        return []
    opts = options_for(
        dataset, booking, actual, services,
        cutoff=acceptance_cutoff(booking, deadline_days),
    )
    return [o.label for o in opts]


def usage_by_cohort(plan: Plan, dataset: Dataset) -> dict[str, dict[tuple, int]]:
    """同一情景内按客群拆分的实际占用，用于说明同一次延期不能按互斥窗口取大。"""
    out: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    for bid, opt in plan.assignments.items():
        if opt is None:
            continue
        b = dataset.bookings[bid]
        for key in resource_keys(opt.slot):
            out[b.cohort][key] += b.people
    return out
