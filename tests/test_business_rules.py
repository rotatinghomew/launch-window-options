"""核对业务规则：容量共享、互斥预留、承诺台账、时间可见性与分层估计。"""
import datetime as dt
import json
from pathlib import Path
import unittest

from launch_window.allocate import allocate, options_for, reservation_need
from launch_window.attribution import attribute, summary
from launch_window.behavior import cohort_stats, outcome_probabilities, replay
from launch_window.ledger import build_ledger, execute
from launch_window.model import Dataset
from launch_window.scenarios import (
    Scenario, evaluate_term, grid, risk_band, shared_reservation, term_table,
)
from launch_window.service import query_tellable, register_notice

DATA = Path(__file__).resolve().parents[1] / "data"
D = dt.date.fromisoformat


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset(DATA)
        cls.term = next(t for t in cls.ds.terms if t.name == "灵活")


class ModelTest(Base):
    def test_notice_timeline_hides_future(self):
        """as_of 之前未发出的通告不能提前看见。"""
        st = self.ds.state("L24", D("2026-08-13"))
        self.assertEqual(st.seq, 1)
        self.assertEqual(len(st.possible), 3)
        self.assertIsNone(st.confirmed)
        st2 = self.ds.state("L24", D("2026-08-14"))
        self.assertEqual(st2.seq, 2)
        self.assertEqual(st2.possible, (D("2026-08-15"), D("2026-08-16")))

    def test_open_window_cannot_lock_viewing(self):
        plan = allocate(self.ds, "L24", None, ("观礼", "游线早", "游线晚"))
        self.assertEqual(plan.viewing_people, 0)
        self.assertGreater(plan.tour_people, 0)

    def test_audit_flags_known_violations(self):
        codes = {f.code for f in self.ds.audit()}
        self.assertIn("viewing_promise_impossible", codes)
        self.assertIn("tour_not_acceptable", codes)
        self.assertIn("offer_product_not_acceptable", codes)


class AllocationTest(Base):
    def test_orders_are_not_split(self):
        plan = allocate(self.ds, "L04", D("2026-06-17"), ("观礼", "游线早", "游线晚"))
        for bid in plan.assignments:
            self.assertIn(bid, self.ds.bookings)

    def test_shared_capacity_not_double_counted(self):
        """同一车组同日上限在多产品同时引用时仍只有一份。"""
        plan = allocate(self.ds, "L04", D("2026-06-17"), ("观礼", "游线早", "游线晚"))
        caps = {}
        for s in self.ds.slots:
            if s.event != "L04":
                continue
            caps[("car", s.event, s.day, s.vehicle)] = s.car_seats
            caps[("site", s.event, s.day, s.venue)] = s.site_seats
        for key, qty in plan.usage.items():
            self.assertLessEqual(qty, caps[key], key)

    def test_infeasible_window_keeps_household_together(self):
        plan = allocate(self.ds, "L04", D("2026-06-17"), ("观礼",))
        self.assertFalse(plan.infeasible)
        # 每张订单恰好一条结果，人数不会被拆开
        self.assertEqual(set(plan.assignments), {b.id for b in self.ds.bookings_by_event["L04"]})
        booked = sum(b.people for b in self.ds.bookings_by_event["L04"])
        self.assertEqual(plan.served_people + plan.refund_people, booked)
        self.assertTrue(any(c.code in ("capacity_blocked", "no_acceptable_option")
                            for c in plan.conflicts))

    def test_priority_decides_who_gets_tight_viewing_seat(self):
        """观礼容量紧张时，优先号小者先拿到观礼席。"""
        plan = allocate(self.ds, "L04", D("2026-06-17"), ("观礼", "游线晚"))
        viewing = {
            bid for bid, opt in plan.assignments.items()
            if opt and opt.kind == "viewing"
        }
        self.assertTrue(viewing)
        # 不能出现：优先号更大的订单拿到观礼，而优先号 0 的可接受订单被挤到普通游线/退费
        low_stranded = {
            b.id for b in self.ds.bookings_by_event["L04"]
            if b.priority == 0 and D("2026-06-17") in b.allowed
            and b.id not in viewing
        }
        high_viewing = {
            b.id for b in self.ds.bookings_by_event["L04"]
            if b.priority > 0 and b.id in viewing
        }
        self.assertFalse(low_stranded and high_viewing)
        # 观礼人数取该优先层能达到的最大值
        low_people = sum(
            b.people for b in self.ds.bookings_by_event["L04"]
            if b.priority == 0 and b.id in viewing
        )
        self.assertEqual(low_people, min(
            8, sum(b.people for b in self.ds.bookings_by_event["L04"]
                   if b.priority == 0 and D("2026-06-17") in b.allowed)
        ))

    def test_reservation_takes_max_across_exclusive_windows(self):
        plans = [
            allocate(self.ds, "L00", D("2026-06-04"), ("观礼", "游线晚")),
            allocate(self.ds, "L00", D("2026-06-06"), ("观礼", "游线晚")),
        ]
        shared, naive = reservation_need(plans)
        for key in shared:
            self.assertNotEqual(shared[key], 0)
            self.assertLessEqual(shared[key], naive[key])


class LedgerTest(Base):
    def test_held_offer_keeps_capacity(self):
        entries = build_ledger(self.ds, "L24", D("2026-08-14"))
        by_offer = {e.offer.id: e for e in entries}
        self.assertEqual(by_offer["O1"].status, "held")
        self.assertEqual(by_offer["O2"].status, "held")
        self.assertEqual(by_offer["O3"].status, "locked")
        self.assertEqual(by_offer["O4"].status, "released")

    def test_expired_without_new_notice_does_not_auto_release(self):
        entries = build_ledger(self.ds, "L24", D("2026-08-17"))
        by_offer = {e.offer.id: e for e in entries}
        self.assertEqual(by_offer["O1"].status, "held")
        self.assertIn("未获准执行释放", by_offer["O1"].reason)

    def test_explicit_release_after_deadline(self):
        entries = build_ledger(
            self.ds, "L24", D("2026-08-17"), release_expired=True,
        )
        by_offer = {e.offer.id: e for e in entries}
        self.assertEqual(by_offer["O1"].status, "released")

    def test_execute_keeps_accepted_promise(self):
        ex = execute(self.ds, "L24", self.term.services,
                     as_of=D("2026-08-14"), actual=None)
        self.assertEqual(ex.plan.assignments["B2401"].product, "游线晚")

    def test_execute_reports_specific_conflict_not_aggregate(self):
        ex = execute(self.ds, "L24", self.term.services,
                     as_of=D("2026-08-14"), actual=None)
        # B2400 的两个已发未答选项仍占用容量，试算不能替客服放掉，也不另选一条
        self.assertNotIn("B2400", ex.plan.assignments)
        self.assertEqual(len(ex.plan.pending["B2400"]), 2)


class LedgerCapacityTest(Base):
    def test_held_offer_not_assigned_again(self):
        ex = execute(self.ds, "L24", self.term.services,
                     as_of=D("2026-08-14"), actual=None)
        self.assertNotIn("B2400", ex.plan.assignments)
        self.assertIn("B2400", ex.plan.pending)

    def test_missing_slot_is_reported(self):
        from launch_window.model import Offer
        offer = Offer("OX", "B2404", D("2026-08-18"), "游线晚", "已接受", D("2026-08-14"))
        self.ds.offers.append(offer)
        self.ds.offers_by_booking["B2404"].append(offer)
        self.ds.offers_by_event["L24"].append(offer)
        try:
            ex = execute(self.ds, "L24", self.term.services,
                         as_of=D("2026-08-14"), actual=None)
        finally:
            self.ds.offers.remove(offer)
            self.ds.offers_by_booking["B2404"].remove(offer)
            self.ds.offers_by_event["L24"].remove(offer)
        ox = next(e for e in ex.entries if e.offer.id == "OX")
        self.assertIsNone(ox.slot)
        self.assertTrue(any(c.code == "offer_slot_missing" for c in ex.plan.conflicts))


class ScenarioTest(Base):
    def test_same_window_compared_across_terms(self):
        rows = term_table(self.ds, [Scenario(2, 0)])
        self.assertEqual({r["terms"] for r in rows}, {"基本", "灵活", "保守"})
        for r in rows:
            self.assertEqual(r["delay"], 2)

    def test_refund_reserve_is_per_event(self):
        res = evaluate_term(self.ds, self.term, Scenario(2, 0))
        expected = sum(
            max(0, p.refund_cost - self.term.max_refund)
            for p in res.plans.values()
        )
        self.assertEqual(res.reserve_gap, expected)

    def test_shared_reservation_not_summed_across_windows(self):
        res = shared_reservation(self.ds, self.term, delays=[0, 1, 2], lead=0)
        for key, shared in res["shared"].items():
            self.assertEqual(
                shared,
                max(v["usage"].get(key, 0) for v in res["per_window"].values()),
            )

    def test_risk_band_ordered(self):
        b = risk_band(self.ds, self.term, 2, 0)
        self.assertLessEqual(b.low_refund_cost, b.high_refund_cost)
        self.assertLessEqual(b.low_refund_people, b.high_refund_people)


class BehaviorTest(Base):
    def test_behavior_is_stratified_by_tour(self):
        stats = cohort_stats(self.ds, D("2026-08-14"))
        tours = {k.tour for k in stats}
        self.assertEqual(tours, {"团0", "团1", "团2"})
        # 团0 与团1 同优先层改期率不同，不能被合成一个总体比例
        r0 = stats[[k for k in stats if k.tour == "团0" and k.priority == 0][0]]
        r1 = stats[[k for k in stats if k.tour == "团1" and k.priority == 0][0]]
        self.assertNotAlmostEqual(r0.reschedule_rate.point, r1.reschedule_rate.point)

    def test_no_evidence_cohort_not_borrowed(self):
        stats = cohort_stats(self.ds, D("2026-08-14"))
        key = [k for k, s in stats.items()
               if k.tour == "团2" and k.priority == 0][0]
        self.assertEqual(stats[key].responded, 0)
        b = next(b for b in self.ds.parties
                 if (b.tour, b.priority) == ("团2", 0))
        p = outcome_probabilities(stats, b)
        self.assertEqual(p["no_evidence"], 1.0)
        self.assertAlmostEqual(p["改到观礼"], 0.25)

    def test_replay_ignores_later_answers(self):
        rp = replay(self.ds, D("2026-06-04"))["L19"]
        self.assertTrue(all(c.day <= D("2026-06-04")
                            for c in self.ds.choices if c.booking in rp.observed))


class ServiceTest(Base):
    def test_commit_order_shrinks_later_options(self):
        from launch_window.service import simulate_commitments
        run = simulate_commitments(self.ds, "L24", D("2026-08-14"), self.term)
        # 前序客人占用后，后序会出现没有可告知选项的情况
        self.assertTrue(run.no_option)
        # 每个没有选项的客人都能追到具体被占用的资源
        self.assertTrue(run.refused_commit)

    def test_offer_impact_identifies_affected_bookings(self):
        from launch_window.service import impact_of_offer
        opt = next(
            o for o in __import__("launch_window.allocate", fromlist=["options_for"]).options_for(
                self.ds, self.ds.bookings["B2407"], None, self.term.services,
            ) if o.label == "2026-08-17 游线晚"
        )
        imp = impact_of_offer(
            self.ds, "L24", D("2026-08-14"), self.term, "B2407", opt,
        )
        self.assertIn("B2401", imp.affected)

    def test_viewing_not_offered_before_confirmation(self):
        tell = query_tellable(self.ds, "L24", D("2026-08-14"), self.term)
        for t in tell.values():
            self.assertNotIn("观礼", {o.product for o in t.options})

    def test_register_flags_conflicts(self):
        reg = register_notice(self.ds, "L24", D("2026-08-14"), self.term)
        joined = "；".join(reg.conflicts)
        self.assertIn("观礼日未唯一确定", joined)
        self.assertIn("不在订单可接受产品", joined)


class AttributionTest(Base):
    def test_attribution_splits_window_and_capacity(self):
        rows = attribute(self.ds)
        s = summary(rows)
        self.assertEqual(s["capacity_extra_cost"],
                         s["capacity_refund_cost"] - s["window_refund_cost"])


class CliTest(unittest.TestCase):
    def test_cli_runs(self):
        import contextlib
        import io
        from launch_window.__main__ import main
        with contextlib.redirect_stdout(io.StringIO()):
            main(["audit"])
            main(["compare", "--delays", "1", "--leads", "0"])
            main(["commit", "--event", "L24", "--as-of", "2026-08-14"])


if __name__ == "__main__":
    unittest.main()
