"""业务资料装载、时间线与资源模型。

本模块只负责把 data/ 中的合成资料转成可复算的对象，并把资料内明确可见的
业务冲突登记下来，不做任何推算。所有时点判断都基于通告的 ``known`` 日：
只有在 ``as_of`` 当日或之前已经发出的通告才算“当时能知道的信息”。
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DATE = dt.date.fromisoformat
PRODUCTS = ("观礼", "游线早", "游线晚")
TOURS = ("游线早", "游线晚")


def _day(value: str) -> dt.date:
    return DATE(value)


@dataclass(frozen=True)
class Notice:
    event: str
    seq: int
    kind: str
    known: dt.date
    possible: tuple[dt.date, ...]
    site: str


@dataclass(frozen=True)
class Booking:
    id: str
    event: str
    first_day: dt.date
    tour: str
    people: int
    allowed: tuple[dt.date, ...]
    fallback: tuple[str, ...]
    refund: int
    priority: int
    until: dt.date
    existing: bool
    option: str

    @property
    def cohort(self) -> tuple:
        """只使用签约时就已知的属性，避免把结果信息带进分层。"""
        return (self.tour, self.priority, self.option, self.fallback)


@dataclass(frozen=True)
class Slot:
    event: str
    day: dt.date
    product: str
    vehicle: str
    venue: str
    car_seats: int
    site_seats: int
    car_cost: int
    site_cost: int

    @property
    def operator_cost(self) -> int:
        return self.car_cost + self.site_cost


@dataclass(frozen=True)
class Offer:
    id: str
    booking: str
    option_day: dt.date
    product: str
    state: str
    sent: dt.date


@dataclass(frozen=True)
class Choice:
    booking: str
    day: dt.date
    offer_day: dt.date
    choice: str
    actual_per_unit: int


@dataclass(frozen=True)
class Terms:
    name: str
    deadline_days: int
    max_refund: int
    services: tuple[str, ...]
    min_notice: int


@dataclass(frozen=True)
class NoticeState:
    event: str
    seq: int
    kind: str
    known: dt.date
    possible: tuple[dt.date, ...]
    site: str
    confirmed: dt.date | None

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed is not None


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warn" | "info"
    code: str
    message: str
    refs: tuple[str, ...] = ()


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class Dataset:
    """全部业务资料及索引。"""

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.root = root
        self.notices = [
            Notice(
                r["event"], int(r["notice"]), r["kind"], _day(r["known"]),
                tuple(_day(x) for x in r["possible_days"]), r["site"],
            )
            for r in _read(root / "notices.json")
        ]
        self.parties = [
            Booking(
                r["booking"], r["event"], _day(r["first_day"]), r["tour"],
                int(r["people"]), tuple(_day(x) for x in r["allowed_days"]),
                tuple(r["fallback"]), int(r["refund"]), int(r["priority"]),
                _day(r["until"]), bool(r["existing"]), r["option"],
            )
            for r in _read(root / "parties.json")
        ]
        self.choices = [
            Choice(r["booking"], _day(r["day"]), _day(r["offer_day"]),
                   r["choice"], int(r["actual_per_unit"]))
            for r in _read(root / "choices.json")
        ]
        self.slots = [
            Slot(
                r["event"], _day(r["day"]), r["product"], r["vehicle"], r["venue"],
                int(r["car_seats"]), int(r["site_seats"]),
                int(r["car_cost"]), int(r["site_cost"]),
            )
            for r in _read(root / "allocations.json")
        ]
        self.offers = [
            Offer(r["offer"], r["booking"], _day(r["option_day"]), r["product"],
                  r["state"], _day(r["sent"]))
            for r in _read(root / "offers.json")
        ]
        self.terms = [
            Terms(r["terms"], int(r["deadline_days"]), int(r["max_refund"]),
                  tuple(r["services"]), int(r["min_notice"]))
            for r in _read(root / "terms.json")
        ]

        self.bookings: dict[str, Booking] = {b.id: b for b in self.parties}
        self.bookings_by_event: dict[str, list[Booking]] = defaultdict(list)
        for b in self.parties:
            self.bookings_by_event[b.event].append(b)
        for rows in self.bookings_by_event.values():
            rows.sort(key=lambda b: (not b.existing, b.priority, b.id))

        self.choices_by_booking = {c.booking: c for c in self.choices}
        self.offers_by_booking: dict[str, list[Offer]] = defaultdict(list)
        for o in self.offers:
            self.offers_by_booking[o.booking].append(o)
        self.offers_by_event: dict[str, list[Offer]] = defaultdict(list)
        for o in self.offers:
            self.offers_by_event[self.bookings[o.booking].event].append(o)

        self.notices_by_event: dict[str, list[Notice]] = defaultdict(list)
        for n in self.notices:
            self.notices_by_event[n.event].append(n)
        for rows in self.notices_by_event.values():
            rows.sort(key=lambda n: (n.known, n.seq))

        self.slots_by_key: dict[tuple[str, dt.date, str], Slot] = {}
        for s in self.slots:
            self.slots_by_key[(s.event, s.day, s.product)] = s

        self.events = sorted(self.notices_by_event)

    # ------------------------------------------------------------------ 时间线
    def state(self, event: str, as_of: dt.date) -> NoticeState:
        """返回 as_of 当日或之前已经发出的最后一条通告。"""
        known = [n for n in self.notices_by_event[event] if n.known <= as_of]
        if not known:
            n = self.notices_by_event[event][0]
            return NoticeState(n.event, 0, "未发出", n.known, (), n.site, None)
        n = known[-1]
        confirmed = n.possible[0] if n.kind == "确认" and len(n.possible) == 1 else None
        return NoticeState(n.event, n.seq, n.kind, n.known, n.possible, n.site, confirmed)

    def final_state(self, event: str) -> NoticeState:
        n = self.notices_by_event[event][-1]
        confirmed = n.possible[0] if n.kind == "确认" and len(n.possible) == 1 else None
        return NoticeState(n.event, n.seq, n.kind, n.known, n.possible, n.site, confirmed)

    def confirmed_day(self, event: str) -> dt.date | None:
        """整个资料期内最终确认的观礼日；未确认返回 None。"""
        for n in reversed(self.notices_by_event[event]):
            if n.kind == "确认" and len(n.possible) == 1:
                return n.possible[0]
        return None

    def initial_possible(self, event: str) -> tuple[dt.date, ...]:
        return self.notices_by_event[event][0].possible

    def notice_lead(self, event: str, as_of: dt.date) -> int | None:
        """最后一条通告的 known 日距其最早可能日（原售日）的天数。"""
        st = self.state(event, as_of)
        if not st.possible:
            return None
        return (st.possible[0] - self.initial_possible(event)[0]).days

    # ------------------------------------------------------------------ 资源
    def slot(self, event: str, day: dt.date, product: str) -> Slot | None:
        return self.slots_by_key.get((event, day, product))

    def event_days(self, event: str) -> list[dt.date]:
        return sorted({s.day for s in self.slots if s.event == event})

    def resources(self, event: str, day: dt.date, product: str):
        """返回该产品当日占用的 (kind, 资源名, 上限, 人均成本)；缺失返回 None。"""
        s = self.slot(event, day, product)
        if s is None:
            return None
        return (
            ("car", s.vehicle, s.car_seats, s.car_cost),
            ("site", s.venue, s.site_seats, s.site_cost),
        )

    # ------------------------------------------------------------------ 审计
    def audit(self) -> list[Finding]:
        out: list[Finding] = []
        add = lambda level, code, msg, refs=(): out.append(Finding(level, code, msg, refs))

        # 通告序列必须逐次收敛，确认通告唯一
        for event, rows in self.notices_by_event.items():
            seen = set()
            prev: Notice | None = None
            for n in rows:
                if n.seq in seen:
                    add("error", "notice_duplicate", f"{event} 通告次序 {n.seq} 重复", (event,))
                seen.add(n.seq)
                if not n.possible:
                    add("error", "notice_empty", f"{event} 通告 {n.seq} 无可观礼日", (event,))
                if n.kind == "确认" and len(n.possible) != 1:
                    add("error", "confirm_not_unique",
                        f"{event} 确认通告未唯一确定观礼日", (event,))
                if prev is not None:
                    if n.known < prev.known:
                        add("error", "notice_time_order",
                            f"{event} 通告 {n.seq} 日期早于前一条", (event,))
                    if not set(n.possible) <= set(prev.possible):
                        add("error", "notice_not_nested",
                            f"{event} 通告 {n.seq} 的可能日不是前一条的子集", (event,))
                prev = n
            if sum(1 for n in rows if n.kind == "确认") > 1:
                add("error", "multiple_confirms", f"{event} 有多条确认通告", (event,))
            if not any(n.kind == "确认" for n in rows):
                add("warn", "window_open", f"{event} 尚未确认实际观礼日", (event,))

        # 订单可接受集合与原售可能日
        for b in self.parties:
            initial = set(self.initial_possible(b.event))
            if not set(b.allowed) <= initial:
                add("warn", "allowed_outside_window",
                    f"{b.id} 可接受日期不在首批可能日内", (b.id,))
            if b.people <= 0:
                add("error", "people_nonpositive", f"{b.id} 人数非正", (b.id,))
            if not b.allowed:
                add("error", "allowed_empty", f"{b.id} 没有可接受日期", (b.id,))
            if any(p not in PRODUCTS for p in b.fallback):
                add("error", "fallback_unknown", f"{b.id} 备用产品不在产品表", (b.id,))
            if b.first_day != self.initial_possible(b.event)[0]:
                add("warn", "first_day_not_initial",
                    f"{b.id} 原售日不是首批最早日", (b.id,))

        # 已结算回复
        for c in self.choices:
            b = self.bookings.get(c.booking)
            if b is None:
                add("error", "choice_orphan", f"{c.booking} 回复无订单", (c.booking,))
                continue
            if c.offer_day not in b.allowed:
                add("error", "choice_day_not_allowed",
                    f"{c.booking} 回复日期不在可接受日期", (c.booking,))
            if c.choice == "同意普通游线":
                if not b.fallback:
                    add("error", "tour_not_acceptable",
                        f"{c.booking} 接受了未列入可接受范围的普通游线", (c.booking,))
                if c.offer_day not in b.allowed:
                    add("error", "tour_day_not_allowed",
                        f"{c.booking} 普通游线日期不在可接受日期", (c.booking,))
            if c.choice == "改到观礼":
                actual = self.confirmed_day(b.event)
                if actual is not None and actual not in b.allowed:
                    add("error", "viewing_promise_impossible",
                        f"{c.booking} 承诺的观礼日在最终窗口中不可接受", (c.booking,))
                if actual is not None and c.offer_day != actual:
                    add("error", "viewing_promise_not_actual",
                        f"{c.booking} 承诺观礼日 {c.offer_day} 不是最终确认日 {actual}",
                        (c.booking,))
                st = self.state(b.event, c.day)
                if c.offer_day not in st.possible:
                    add("warn", "viewing_promise_not_possible_then",
                        f"{c.booking} 回复当日 {c.offer_day} 已不在可能窗口内",
                        (c.booking,))
            if c.choice == "明确退费" and c.actual_per_unit != b.refund:
                add("warn", "refund_amount_differs",
                    f"{c.booking} 已结退费与合同退费不同", (c.booking,))
            if c.day < b.first_day or c.day > b.until:
                add("warn", "response_out_of_window",
                    f"{c.booking} 回复日不在订单有效期内", (c.booking,))

        # 已发出的备选单
        for o in self.offers:
            b = self.bookings.get(o.booking)
            if b is None:
                add("error", "offer_orphan", f"{o.id} 告知单无订单", (o.id,))
                continue
            if o.option_day not in b.allowed:
                add("error", "offer_day_not_allowed",
                    f"{o.id} 告知日期不在订单可接受日期", (o.id,))
            if o.product == "观礼":
                st = self.state(b.event, o.sent)
                if not st.is_confirmed and st.confirmed != o.option_day:
                    add("warn", "viewing_offer_before_confirm",
                        f"{o.id} 在观礼日未唯一确定前发出观礼选项", (o.id,))
                elif st.confirmed and st.confirmed != o.option_day:
                    add("error", "viewing_offer_stale",
                        f"{o.id} 观礼选项与已确认日不一致", (o.id,))
            elif o.product not in b.fallback:
                add("error", "offer_product_not_acceptable",
                    f"{o.id} {o.product} 不在订单可接受产品", (o.id,))
            if o.sent > b.until:
                add("error", "offer_after_deadline",
                    f"{o.id} 发出日已过订单回复截止", (o.id,))
            if o.state == "已接受":
                st = self.state(b.event, o.sent)
                if o.product == "观礼" and st.confirmed != o.option_day:
                    add("warn", "accepted_viewing_unconfirmed",
                        f"{o.id} 接受的观礼日当时尚未确认", (o.id,))

        # 运力：同车同日、同点位同日跨产品只能有一份上限
        car_caps: dict[tuple, set] = defaultdict(set)
        site_caps: dict[tuple, set] = defaultdict(set)
        for s in self.slots:
            car_caps[(s.event, s.day, s.vehicle)].add(s.car_seats)
            site_caps[(s.event, s.day, s.venue)].add(s.site_seats)
        for key, caps in car_caps.items():
            if len(caps) > 1:
                add("error", "car_cap_conflict", f"{key} 同车上限不一致 {sorted(caps)}")
        for key, caps in site_caps.items():
            if len(caps) > 1:
                add("error", "site_cap_conflict", f"{key} 同点位上限不一致 {sorted(caps)}")
        keys = [(s.event, s.day, s.product) for s in self.slots]
        if len(keys) != len(set(keys)):
            add("error", "slot_duplicate", "同一产品同日存在多条运力记录")

        # 条款
        for t in self.terms:
            if not t.services or not set(t.services) <= set(PRODUCTS):
                add("error", "terms_services", f"{t.name} 条款服务集合非法")
            if t.deadline_days < 0 or t.min_notice < 0:
                add("error", "terms_negative", f"{t.name} 条款天数为负")
            if "观礼" not in t.services:
                add("warn", "terms_no_viewing", f"{t.name} 条款不含观礼")
            for event in self.events:
                exposure = sum(b.people * b.refund for b in self.bookings_by_event[event])
                if exposure > t.max_refund:
                    add("info", "reserve_below_exposure",
                        f"{t.name} 在 {event} 的退费准备金低于全额退费敞口",
                        (t.name, event))
        return out


def print_findings(findings: Iterable[Finding]) -> None:
    for f in findings:
        refs = f" ({', '.join(f.refs)})" if f.refs else ""
        print(f"[{f.level:5}] {f.code}: {f.message}{refs}")
