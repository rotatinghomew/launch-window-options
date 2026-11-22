"""容量可行的行程分配与承诺台账。

分配模型把每位订单一同安置（不可拆），只使用订单可接受日期和产品；同一车组、
同一接待区在同一天只有一份人数上限，多个产品同时引用时也共用这份上限。

承诺台账单独记录“已经说出去的选项”：已接受视为锁定，已发未答在回复前继续占用
容量，已拒绝才释放。这样重算不会把尚有效的承诺悄悄放掉。
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import Booking, Dataset, Slot

REFUND = "退费"
VIEWING = "观礼"


@dataclass(frozen=True)
class Option:
    day: dt.date
    product: str
    kind: str  # "viewing" | "tour"
    slot: Slot

    @property
    def label(self) -> str:
        return f"{self.day.isoformat()} {self.product}"


@dataclass(frozen=True)
class Conflict:
    booking: str
    code: str
    message: str


@dataclass
class Plan:
    event: str
    actual: dt.date | None
    assignments: dict[str, Option | None] = field(default_factory=dict)
    usage: dict[tuple, int] = field(default_factory=dict)
    reserved: dict[tuple, int] = field(default_factory=dict)
    served_people: int = 0
    pending_people: int = 0
    refund_people: int = 0
    viewing_people: int = 0
    tour_people: int = 0
    operator_cost: int = 0
    passenger_cost: int = 0
    refund_cost: int = 0
    conflicts: list[Conflict] = field(default_factory=list)
    pending: dict[str, list[Option]] = field(default_factory=dict)
    infeasible: bool = False

    @property
    def total_cost(self) -> int:
        return self.operator_cost + self.passenger_cost + self.refund_cost

    @property
    def served_bookings(self) -> int:
        return sum(1 for a in self.assignments.values() if a is not None)

    @property
    def refund_bookings(self) -> int:
        return sum(1 for a in self.assignments.values() if a is None)

    def rows(self) -> list[dict]:
        out = []
        for bid in sorted(self.assignments):
            opt = self.assignments[bid]
            out.append({
                "booking": bid,
                "day": opt.day.isoformat() if opt else None,
                "product": opt.product if opt else REFUND,
                "kind": opt.kind if opt else "refund",
            })
        return out


def resource_keys(slot: Slot) -> tuple[tuple, tuple]:
    car = ("car", slot.event, slot.day, slot.vehicle)
    site = ("site", slot.event, slot.day, slot.venue)
    return car, site


def person_operator_cost(slot: Slot) -> int:
    return slot.car_cost + slot.site_cost


def acceptance_cutoff(booking: Booking, deadline_days: int | None = None) -> dt.date:
    """订单最晚接受日；新条款只能收紧，不能越过订单自身的回复截止。"""
    cutoff = booking.until
    if deadline_days is not None:
        cutoff = min(cutoff, booking.first_day + dt.timedelta(days=deadline_days))
    return cutoff


def options_for(
    dataset: Dataset,
    booking: Booking,
    actual: dt.date | None,
    services: Sequence[str],
    *,
    cutoff: dt.date | None = None,
    allow_viewing: bool = True,
    exchange_available: bool = True,
) -> list[Option]:
    """列出订单在给定窗口结果下真实可接受的选项。

    区分三层权利：

    * **可以先售的权利**：原售日的观礼票；按订单约定可独立锁定的普通游线。
      这些不依赖新的换期通告。
    * **通告明确后才发出的换期选项**：改到新的观礼日。只有在正式通告唯一确定
      观礼日、且通告提前量满足条款最少通知日时才可用。
    * **需要退费的部分**：以上都不能兑现时按合同退费。

    观礼日未唯一确定（``actual is None``）时不生成任何观礼选项。
    """
    cutoff = cutoff or booking.until
    out: list[Option] = []
    if allow_viewing and actual is not None and VIEWING in services:
        if actual in booking.allowed and actual <= cutoff:
            # 留在原售日属于先售权利；改到别的日子才需要通告后的换期选项
            if actual == booking.first_day or exchange_available:
                slot = dataset.slot(booking.event, actual, VIEWING)
                if slot:
                    out.append(Option(actual, VIEWING, "viewing", slot))
    for product in services:
        if product == VIEWING or product not in booking.fallback:
            continue
        for day in booking.allowed:
            if day > cutoff:
                continue
            slot = dataset.slot(booking.event, day, product)
            if slot:
                out.append(Option(day, product, "tour", slot))
    return out


def _capacities(dataset: Dataset, event: str) -> dict[tuple, int]:
    caps: dict[tuple, int] = {}
    for s in dataset.slots:
        if s.event != event:
            continue
        car, site = resource_keys(s)
        caps[car] = s.car_seats
        caps[site] = s.site_seats
    return caps


def _add_usage(usage: dict[tuple, int], slot: Slot, people: int) -> None:
    for key in resource_keys(slot):
        usage[key] = usage.get(key, 0) + people


def allocate(
    dataset: Dataset,
    event: str,
    actual: dt.date | None,
    services: Sequence[str],
    *,
    deadline_days: int | None = None,
    allow_viewing: bool = True,
    exchange_available: bool = True,
    ignore_capacity: bool = False,
    excluded: Iterable[str] = (),
    locked: Mapping[str, Option] | None = None,
    reserved: Iterable[tuple[str, Option, int]] = (),
    force_refund: Iterable[str] = (),
) -> Plan:
    """在指定窗口结果下求解容量可行的分配。

    参数
    ----
    actual: 实际观礼日；尚未确认时传 None，此时不产生观礼选项。
    services: 本条款可提供的产品。
    deadline_days: 新条款允许的原定日后最晚接受天数。
    excluded: 已释放、不再参与本次分配的订单。
    locked: 已接受并锁定的选项。
    reserved: 已发出、尚未答复的选项及其人数，先占容量。
    force_refund: 风险最坏情形下强制退费的订单。
    """
    locked = dict(locked or {})
    excluded = set(excluded)
    force_refund = set(force_refund)
    bookings = dataset.bookings_by_event[event]
    caps = _capacities(dataset, event)
    plan = Plan(event=event, actual=actual)

    used: dict[tuple, int] = defaultdict(int)
    reserved_used: dict[tuple, int] = defaultdict(int)
    for bid, opt, people in reserved:
        _add_usage(reserved_used, opt.slot, people)
    for bid, opt in locked.items():
        b = dataset.bookings[bid]
        if bid not in bookings_by_event_ids(bookings):
            continue
        # 已接受是已发出的权益，不因新条款的服务清单缺失而失效；
        # 但日期必须在订单可接受范围内，观礼日必须确已唯一确定且与锁定日一致。
        problem = _locked_problem(dataset, b, opt, actual, allow_viewing)
        if problem:
            plan.conflicts.append(Conflict(
                bid, "locked_option_invalid",
                f"{bid} 已锁定的 {opt.label} 在当前窗口下不可兑现：{problem}",
            ))
            continue
        _add_usage(used, opt.slot, b.people)
        plan.assignments[bid] = opt

    # 固定占用（锁定 + 已发未答）超出上限时，仍继续求剩余分配，但显式记录冲突
    for key, cap in caps.items():
        need = used.get(key, 0) + reserved_used.get(key, 0)
        if need > cap:
            plan.infeasible = True
            plan.conflicts.append(Conflict(
                "", "capacity_overcommit",
                f"{key[0]} {key[2].isoformat()} {key[3]} 已占用 {need} 人，上限 {cap} 人",
            ))

    # 已有未答告知的订单由预留占位，本次试算不再替它另选一条
    pending: dict[str, list[Option]] = defaultdict(list)
    for bid, opt, people in reserved:
        pending[bid].append(opt)
    plan.pending = dict(pending)

    free = [
        b for b in bookings
        if b.id not in locked and b.id not in excluded and b.id not in pending
    ]
    if not free:
        plan.usage = {k: used.get(k, 0) for k in set(used) | set(reserved_used)}
        _finalize(dataset, plan, bookings, used, reserved_used)
        return plan

    # 变量：每个订单的可用选项，外加退费
    variables: list[tuple[str, Option | None]] = []
    index: dict[tuple[str, str], int] = {}
    bcache: dict[str, list[Option]] = {}
    for b in free:
        opts = options_for(
            dataset, b, actual, services,
            cutoff=acceptance_cutoff(b, deadline_days), allow_viewing=allow_viewing,
            exchange_available=exchange_available,
        )
        bcache[b.id] = opts
        for opt in opts:
            index[(b.id, opt.label)] = len(variables)
            variables.append((b.id, opt))
        index[(b.id, REFUND)] = len(variables)
        variables.append((b.id, None))
    n = len(variables)
    if n == 0:
        _finalize(dataset, plan, bookings, used, reserved_used)
        return plan

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    # 每个订单恰选一项（服务或退费）
    for b in free:
        row = np.zeros(n)
        for opt in bcache[b.id]:
            row[index[(b.id, opt.label)]] = 1
        row[index[(b.id, REFUND)]] = 1
        rows.append(row)
        lo.append(1)
        hi.append(1)
    if force_refund:
        for b in free:
            if b.id in force_refund:
                row = np.zeros(n)
                row[index[(b.id, REFUND)]] = 1
                rows.append(row)
                lo.append(1)
                hi.append(1)

    # 共享容量：锁定与已发未答先占，剩余给本次分配
    if not ignore_capacity:
        for key, cap in caps.items():
            row = np.zeros(n)
            for j, (bid, opt) in enumerate(variables):
                if opt is not None and key in resource_keys(opt.slot):
                    row[j] = dataset.bookings[bid].people
            rows.append(row)
            lo.append(-np.inf)
            hi.append(cap - used.get(key, 0) - reserved_used.get(key, 0))

    # 严格按（既有订单，优先号）做字典序，共 6 个客群档：
    #   (既有,0)(既有,1)(既有,2)(新,0)(新,1)(新,2)
    # 对每一档依次最大化：
    #   a) 该档拿到的观礼席人数（优先号小者先拿到紧张的观礼位）；
    #   b) 该档被服务的人数（观礼或普通游线），作为该档的兜底。
    # 上一档的最优值固定为下界，后档不能挤占。最后再压低运营与退费成本。
    groups = [(True, 0), (True, 1), (True, 2), (False, 0), (False, 1), (False, 2)]

    def group_row(existing: bool, priority: int, kind: str) -> np.ndarray:
        row = np.zeros(n)
        for j, (bid, opt) in enumerate(variables):
            if opt is None:
                continue
            b = dataset.bookings[bid]
            if b.existing != existing or b.priority != priority:
                continue
            if kind == "viewing" and opt.kind != "viewing":
                continue
            if kind == "presale" and not (opt.kind == "viewing" and opt.day == b.first_day):
                continue
            row[j] = b.people
        return row

    def solve(objective: np.ndarray, fixed: list[tuple[np.ndarray, float]]):
        """``fixed`` 是已完成层的下界：小于它的解一律不可行。"""
        rr = rows + [r for r, _ in fixed]
        ll = lo + [v for _, v in fixed]
        hh = hi + [np.inf for _, v in fixed]
        return milp(objective, constraints=LinearConstraint(np.vstack(rr), ll, hh),
                    integrality=np.ones(n), bounds=Bounds(0, 1))

    def maximise(row: np.ndarray, fixed: list[tuple[np.ndarray, float]]):
        """最大化一个非负行；返回 (解, 该行取值, 是否成功)。"""
        res = solve(-row, fixed)
        ok = bool(res.success)
        value = float(row @ res.x) if ok else 0.0
        return res, value, ok

    def pin(row: np.ndarray, value: float) -> tuple[np.ndarray, float]:
        """把已完成层的最优值固定为下界。

        行值可达 1e7 量级，求解器的可行性容差是相对的；若只减 1e-6，
        下一层可以把 8e6 回落到 7e6 仍被判为“可行”。这里按相对量留余量，
        但余量小于最小权重 1，保证不会真的丢掉一层的最小可行单位。
        """
        if value <= 0:
            return (row, -1e-6)
        return (row, value - max(1e-6, abs(value) * 1e-9))

    fixed: list[tuple[np.ndarray, float]] = []
    last = None

    # 先售出的原售日观礼票是已售权利，整体优先保住
    presale = np.zeros(n)
    for j, (bid, opt) in enumerate(variables):
        if opt is not None and opt.kind == "viewing" and opt.day == dataset.bookings[bid].first_day:
            presale[j] = dataset.bookings[bid].people
    res, v, ok = maximise(presale, [])
    if not ok:
        plan.infeasible = True
        plan.conflicts.append(Conflict("", "milp_infeasible", "当前约束下无可行分配"))
        _finalize(dataset, plan, bookings, used, reserved_used)
        return plan
    fixed.append(pin(presale, v))
    last = res

    for existing, priority in groups:
        for kind in ("viewing", "service"):
            row = group_row(existing, priority, kind)
            res, v, ok = maximise(row, fixed)
            if ok:
                fixed.append(pin(row, v))
                last = res

    # 5) 成本
    costs = np.zeros(n)
    for j, (bid, opt) in enumerate(variables):
        b = dataset.bookings[bid]
        if opt is None:
            costs[j] = b.people * b.refund
        else:
            costs[j] = b.people * person_operator_cost(opt.slot)
    costs = costs + np.arange(n) * 1e-9
    res4 = solve(costs, fixed)
    final = res4 if res4.success else last

    for j, (bid, opt) in enumerate(variables):
        if final.x[j] > 0.5:
            plan.assignments[bid] = opt
            if opt is not None:
                _add_usage(used, opt.slot, dataset.bookings[bid].people)

    _finalize(dataset, plan, bookings, used, reserved_used)
    _diagnose(dataset, plan, bookings, bcache, caps, used, reserved_used)
    return plan


def bookings_by_event_ids(bookings: Sequence[Booking]) -> set[str]:
    return {b.id for b in bookings}


def _locked_problem(
    dataset: Dataset,
    booking: Booking,
    opt: Option,
    actual: dt.date | None,
    allow_viewing: bool,
) -> str | None:
    """已接受承诺能否兑现；返回具体原因，能兑现返回 None。"""
    if opt.day not in booking.allowed:
        return f"日期 {opt.day.isoformat()} 不在订单可接受日期"
    if opt.product != VIEWING and opt.product not in booking.fallback:
        return f"产品 {opt.product} 不在订单可接受产品"
    if opt.product == VIEWING:
        if not allow_viewing:
            return "条款不含观礼"
        if actual is None:
            return "观礼日尚未由正式通告唯一确定"
        if opt.day != actual:
            return f"已确认观礼日为 {actual.isoformat()}，与锁定日不一致"
    return None


def _finalize(
    dataset: Dataset,
    plan: Plan,
    bookings: Sequence[Booking],
    used: dict[tuple, int],
    reserved_used: dict[tuple, int],
) -> None:
    plan.usage = {k: used.get(k, 0) for k in set(used) | set(reserved_used)}
    plan.reserved = dict(reserved_used)
    plan.pending_people = sum(
        dataset.bookings[bid].people for bid in plan.pending
    )
    for b in bookings:
        opt = plan.assignments.get(b.id)
        if opt is None:
            continue
        plan.served_people += b.people
        if opt.kind == "viewing":
            plan.viewing_people += b.people
        else:
            plan.tour_people += b.people
        plan.operator_cost += b.people * person_operator_cost(opt.slot)
    for b in bookings:
        if b.id in plan.assignments and plan.assignments[b.id] is None:
            plan.refund_people += b.people
            plan.refund_cost += b.people * b.refund


def _diagnose(
    dataset: Dataset,
    plan: Plan,
    bookings: Sequence[Booking],
    bcache: Mapping[str, list[Option]],
    caps: Mapping[tuple, int],
    used: Mapping[tuple, int],
    reserved_used: Mapping[tuple, int],
) -> None:
    """为未兑现的订单登记具体冲突和仍可保留的替代。"""
    residual = {k: caps[k] - used.get(k, 0) - reserved_used.get(k, 0) for k in caps}
    for b in bookings:
        if plan.assignments.get(b.id, "absent") is not None:
            continue
        opts = bcache.get(b.id, [])
        if not opts:
            plan.conflicts.append(Conflict(
                b.id, "no_acceptable_option",
                f"{b.id} 在当前窗口和条款下没有可接受的服务，只能退费 {b.people} 人",
            ))
            continue
        alt = []
        blocked = []
        for opt in opts:
            keys = resource_keys(opt.slot)
            slack = min(residual.get(k, 0) for k in keys)
            if slack >= b.people:
                alt.append(opt.label)
            else:
                tight = min(keys, key=lambda k: residual.get(k, 0))
                blocked.append((opt.label, tight, residual.get(tight, 0), caps.get(tight, 0)))
        if alt:
            plan.conflicts.append(Conflict(
                b.id, "capacity_shortfall",
                f"{b.id} 未能按优先次序兑现；仍可保留的替代：{'；'.join(alt)}",
            ))
        else:
            detail = "；".join(
                f"{label} 需 {b.people} 人，{key[0]} {key[2].isoformat()} {key[3]} "
                f"余 {slack}/{cap}"
                for label, key, slack, cap in blocked
            )
            plan.conflicts.append(Conflict(
                b.id, "capacity_blocked",
                f"{b.id} 所有可接受选项均被占满：{detail}",
            ))


def reservation_need(plans: Sequence[Plan]) -> tuple[dict[tuple, int], dict[tuple, int]]:
    """互斥窗口情景下的共享预留。

    同一次延期只兑现一个实际窗口，因此同一资源键按各情景需求的最大值预留；
    同一情景内不同客群可能同时接受，需求在该情景内已经相加，不再按客群取大。
    返回 (共享预留, 逐情景简单相加) 便于核对不能双算。
    """
    shared: dict[tuple, int] = {}
    naive: dict[tuple, int] = {}
    for plan in plans:
        for key, qty in plan.usage.items():
            shared[key] = max(shared.get(key, 0), qty)
            naive[key] = naive.get(key, 0) + qty
    return shared, naive


def plan_breakdown(plan: Plan) -> dict:
    return {
        "event": plan.event,
        "actual": plan.actual.isoformat() if plan.actual else None,
        "served_people": plan.served_people,
        "pending_people": plan.pending_people,
        "viewing_people": plan.viewing_people,
        "tour_people": plan.tour_people,
        "refund_people": plan.refund_people,
        "operator_cost": plan.operator_cost,
        "refund_cost": plan.refund_cost,
        "total_cost": plan.total_cost,
        "infeasible": plan.infeasible,
        "conflicts": [c.__dict__ for c in plan.conflicts],
    }
