"""命令行入口：核资料、看时间线、比条款、执行承诺、登记与事后回看。

示例
----
    python3 -m launch_window audit
    python3 -m launch_window timeline --as-of 2026-08-14
    python3 -m launch_window compare --delays 0,1,2 --leads 0,1
    python3 -m launch_window reserve --terms 灵活
    python3 -m launch_window execute --event L24 --as-of 2026-08-14
    python3 -m launch_window register --terms 灵活 --event L24 --as-of 2026-08-14
    python3 -m launch_window attribution
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .allocate import plan_breakdown
from .attribution import attribute, summary as attribution_summary
from .behavior import per_event_table, replay
from .ledger import execute
from .model import Dataset, print_findings
from .scenarios import (
    Scenario, behavior_estimate, cost_distribution, evaluate_term,
    expected_behavior_shift, grid, open_window_risk, risk_band,
    shared_reservation, term_table,
)
from .service import (
    impact_of_offer, query_tellable, register_notice, simulate_commitments,
)

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _term(dataset: Dataset, name: str):
    for t in dataset.terms:
        if t.name == name:
            return t
    raise SystemExit(f"没有条款 {name}；可选：{', '.join(t.name for t in dataset.terms)}")


def _print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def cmd_audit(ds: Dataset, args) -> None:
    findings = ds.audit()
    print_findings(findings)
    counts = {}
    for f in findings:
        counts[f.level] = counts.get(f.level, 0) + 1
    print()
    print("汇总：", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "无问题")


def cmd_timeline(ds: Dataset, args) -> None:
    as_of = _date(args.as_of)
    rows = per_event_table(replay(ds, as_of))
    for r in rows:
        r["possible"] = ",".join(r.pop("possible_days")) or "-"
        r["confirmed"] = r["confirmed"] or "未确定"
        r["unresolved"] = r.pop("unresolved_people")
        r.pop("bookings")
    _print_table(rows, [
        "event", "seq", "kind", "known", "possible", "confirmed", "delay",
        "people", "answered_people", "unresolved", "reschedule_people",
        "refund_people",
    ])


def _scenarios(args) -> list[Scenario]:
    delays = [int(x) for x in args.delays.split(",")]
    leads = [int(x) for x in args.leads.split(",")]
    return grid(delays, leads)


def cmd_compare(ds: Dataset, args) -> None:
    rows = term_table(ds, _scenarios(args))
    cols = ["scenario", "terms", "sold_people", "moved_people", "moved_share",
            "viewing_people", "viewing_rebooked_people", "tour_people",
            "refund_people", "refund_cost", "operator_cost", "total_cost",
            "reserve_gap", "conflicts"]
    _print_table(rows, cols)


def cmd_reserve(ds: Dataset, args) -> None:
    term = _term(ds, args.terms)
    delays = [int(x) for x in args.delays.split(",")]
    lead = int(args.lead)
    res = shared_reservation(ds, term, delays=delays, lead=lead)
    shared, naive = res["shared"], res["naive_sum"]
    print(f"条款 {term.name}，延期候选 {delays}，提前 {lead} 天")
    print(f"共享预留资源键 {len(shared)} 个（车组与点位各占一键）")
    total_shared = sum(shared.values())
    total_naive = sum(naive.values())
    print(f"占用键值合计（人·键，同一人会在车组和点位各计一次）："
          f"共享 {total_shared}，简单相加 {total_naive}，双算 {total_naive - total_shared}")
    print("共享口径：同一资源键取各互斥窗口需求的最大值；"
          "同一窗口内不同客群的需求已经相加。")
    print()
    print("发生双算的资源（简单相加 - 共享）：")
    for key, extra in sorted(res["overstated"].items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {key[0]} {key[1]} {key[2].isoformat()} {key[3]}: +{extra}")
    print()
    print("同一窗口内不同客群同时接受的情况（已相加，不再取大）：")
    for d, info in res["per_window"].items():
        multi = {k: v for k, v in info["cohorts"].items() if len(v) > 1}
        print(f"  延期 {d} 天：{len(multi)} 个资源键有多客群共用")
    if args.json:
        print(json.dumps({
            "shared": {str(k): v for k, v in shared.items()},
            "naive": {str(k): v for k, v in naive.items()},
        }, ensure_ascii=False, indent=2))


def cmd_risk(ds: Dataset, args) -> None:
    delays = [int(x) for x in args.delays.split(",")]
    lead = int(args.lead)
    rows = []
    for term in ds.terms:
        for d in delays:
            b = risk_band(ds, term, d, lead)
            rows.append({
                "terms": term.name,
                "delay": b.delay,
                "lead": b.lead,
                "refund_people_low": b.low_refund_people,
                "refund_people_high": b.high_refund_people,
                "refund_cost_low": b.low_refund_cost,
                "refund_cost_high": b.high_refund_cost,
                "total_cost_low": b.low_total_cost,
                "total_cost_high": b.high_total_cost,
                "reserve_gap_high": b.reserve_gap_high,
            })
    _print_table(rows, list(rows[0]))
    print()
    for term in ds.terms:
        b = risk_band(ds, term, delays[-1], lead)
        print(f"{term.name}：{b.reasons[0]}；{b.reasons[1]}；{b.reasons[2]}")
    if args.open:
        print()
        for term in ds.terms:
            r = open_window_risk(ds, term, delays, lead)
            print(f"{term.name} 窗口未定：退费 {r['refund_cost_low']}~{r['refund_cost_high']}，"
                  f"缺口 {r['reserve_gap_high']}；{r['note']}")


def cmd_execute(ds: Dataset, args) -> None:
    term = _term(ds, args.terms)
    as_of = _date(args.as_of)
    st = ds.state(args.event, as_of)
    actual = st.confirmed
    ex = execute(
        ds, args.event, term.services, as_of=as_of, actual=actual,
        deadline_days=term.deadline_days,
        release_expired=args.release_expired,
    )
    print(f"{args.event} 截至 {as_of} 通告 {st.seq}（{st.kind}），"
          f"观礼日 {'未确定' if actual is None else actual}")
    print()
    for e in ex.entries:
        print(f"  {e.offer.id} {e.booking.id} {e.offer.product} "
              f"{e.offer.option_day} -> {e.status}：{e.reason}")
    print()
    print("本次释放：", "无" if not ex.released else "")
    for e in ex.released:
        print(f"  {e.offer.id} {e.booking.id}：{e.reason}")
    print()
    if ex.plan:
        print(json.dumps(plan_breakdown(ex.plan), ensure_ascii=False, indent=2))


def cmd_register(ds: Dataset, args) -> None:
    term = _term(ds, args.terms)
    as_of = _date(args.as_of)
    reg = register_notice(ds, args.event, as_of, term)
    print(f"{args.event} 截至 {as_of} 通告 {reg.state.seq}（{reg.state.kind}），"
          f"可能日 {[d.isoformat() for d in reg.state.possible]}")
    print("已发出：", len(reg.sent), "，已接受：", len(reg.accepted),
          "，已发未答：", len(reg.held))
    for c in reg.conflicts:
        print("  冲突：", c)
    print()
    for bid, t in reg.tellable.items():
        print(f"  {bid} {t.reason}：{'；'.join(t.labels) if t.labels else '无'}")
        for b in t.blocked:
            print(f"      被占用：{b}")


def cmd_impact(ds: Dataset, args) -> None:
    term = _term(ds, args.terms)
    as_of = _date(args.as_of)
    b = ds.bookings[args.booking]
    st = ds.state(args.event, as_of)
    tell = query_tellable(ds, args.event, as_of, term).get(args.booking)
    if tell is None or not tell.options:
        print(f"{args.booking} 在 {args.event} 截至 {as_of} 无可告知选项")
        return
    for opt in tell.options:
        imp = impact_of_offer(ds, args.event, as_of, term, args.booking, opt)
        print(f"若把 {opt.label} 说给 {args.booking}：")
        if not imp.affected:
            print("  对其他客人的可告知范围没有影响")
        else:
            print(f"  影响 {len(imp.affected)} 位客人："
                  + "；".join(f"{k} 少 {v} 项" for k, v in sorted(imp.affected.items())))
            if imp.lost_all:
                print(f"  其中 {', '.join(imp.lost_all)} 会完全没有可告知选项")


def cmd_commit(ds: Dataset, args) -> None:
    term = _term(ds, args.terms)
    as_of = _date(args.as_of)
    run = simulate_commitments(
        ds, args.event, as_of, term, prefer=args.prefer,
    )
    print(f"{args.event} 截至 {as_of}，按既有订单优先、再按优先号依次给选项：")
    held = {o.booking for o in ds.offers_by_event.get(args.event, [])
            if o.sent <= as_of and o.state != "已拒绝"}
    for bid in run.order:
        if bid in run.assigned and run.assigned[bid] is not None:
            print(f"  {bid}: {run.assigned[bid].label}")
        elif bid in held:
            print(f"  {bid}: 已有发出的选项，仍占用，暂不重复发")
        else:
            print(f"  {bid}: 无可告知选项，只能按合同退费")
    print()
    print(f"共 {len(run.no_option)} 位客人没有可告知选项：{', '.join(run.no_option) or '无'}")
    if run.refused_commit:
        print("具体被占用的资源：")
        for bid, why in run.refused_commit:
            print(f"  {bid}: {why}")


def cmd_behavior(ds: Dataset, args) -> None:
    as_of = _date(args.as_of)
    rows = behavior_estimate(ds, as_of, tour=args.tour, event=args.event)
    _print_table(rows, ["cohort", "bookings", "people", "responded",
                        "reschedule_n", "reschedule_rate", "reschedule_ci_low",
                        "reschedule_ci_high", "tour_n", "refund_n",
                        "no_reply_n", "evidence"])
    print()
    print("只按销售组与签约属性分层；无回复样本的层不给点估计，"
          "不把其它销售组的改期率套过来。")
    print()
    est = expected_behavior_shift(
        ds, as_of, _term(ds, args.terms), Scenario(int(args.delay), int(args.lead)),
    )
    print(f"把历史分层回复率套到 {est['scenario']}：")
    print(f"  可改期比例 {est['reschedule_share_low']:.3f} ~ {est['reschedule_share_high']:.3f}"
          f"（无样本层 {est['no_evidence_people']} 人；窗口未定 {est['open_window_people']} 人）")
    print(f"  普通游线 {est['tour_share_point']:.3f}，退费 {est['refund_share_point']:.3f}，"
          f"无应答 {est['no_reply_share_point']:.3f}")


def cmd_attribution(ds: Dataset, args) -> None:
    rows = attribute(ds)
    _print_table(
        [r.__dict__ for r in rows],
        ["event", "actual", "observed_people", "observed_cost",
         "window_refund_people", "window_refund_cost",
         "capacity_refund_people", "capacity_refund_cost",
         "promised_viewing_people", "promised_viewing_mismatch_people",
         "promised_tour_people", "promised_unacceptable_people"],
    )
    print()
    print(json.dumps(attribution_summary(rows), ensure_ascii=False, indent=2))


def cmd_costs(ds: Dataset, args) -> None:
    term = _term(ds, args.terms)
    sc = Scenario(int(args.delay), int(args.lead))
    res = evaluate_term(ds, term, sc)
    print(f"{term.name} / {sc.name}")
    print(json.dumps(cost_distribution(res), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="launch_window", description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("audit").set_defaults(func=cmd_audit)

    t = sub.add_parser("timeline")
    t.add_argument("--as-of", required=True)
    t.set_defaults(func=cmd_timeline)

    c = sub.add_parser("compare")
    c.add_argument("--delays", default="0,1,2")
    c.add_argument("--leads", default="0,1")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("reserve")
    r.add_argument("--terms", required=True)
    r.add_argument("--delays", default="0,1,2")
    r.add_argument("--lead", default="0")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_reserve)

    k = sub.add_parser("risk")
    k.add_argument("--delays", default="0,1,2")
    k.add_argument("--lead", default="0")
    k.add_argument("--open", action="store_true")
    k.set_defaults(func=cmd_risk)

    e = sub.add_parser("execute")
    e.add_argument("--event", required=True)
    e.add_argument("--as-of", required=True)
    e.add_argument("--terms", default="灵活")
    e.add_argument("--release-expired", action="store_true")
    e.set_defaults(func=cmd_execute)

    g = sub.add_parser("register")
    g.add_argument("--event", required=True)
    g.add_argument("--as-of", required=True)
    g.add_argument("--terms", default="灵活")
    g.set_defaults(func=cmd_register)

    i = sub.add_parser("impact")
    i.add_argument("--event", required=True)
    i.add_argument("--booking", required=True)
    i.add_argument("--as-of", required=True)
    i.add_argument("--terms", default="灵活")
    i.set_defaults(func=cmd_impact)

    m = sub.add_parser("commit")
    m.add_argument("--event", required=True)
    m.add_argument("--as-of", required=True)
    m.add_argument("--terms", default="灵活")
    m.add_argument("--prefer", choices=["first", "cheapest"], default="first")
    m.set_defaults(func=cmd_commit)

    b = sub.add_parser("behavior")
    b.add_argument("--as-of", required=True)
    b.add_argument("--tour", default=None)
    b.add_argument("--event", default=None)
    b.add_argument("--terms", default="灵活")
    b.add_argument("--delay", default="2")
    b.add_argument("--lead", default="0")
    b.set_defaults(func=cmd_behavior)

    a = sub.add_parser("attribution")
    a.set_defaults(func=cmd_attribution)

    s = sub.add_parser("costs")
    s.add_argument("--terms", required=True)
    s.add_argument("--delay", default="2")
    s.add_argument("--lead", default="0")
    s.set_defaults(func=cmd_costs)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ds = Dataset(args.data)
    args.func(ds, args)


if __name__ == "__main__":
    main()
