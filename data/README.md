# 资料字典

所有费用单位为人民币元，时间为北京时间自然日。正式通告唯一确定观礼日后方可锁定相应的观礼席位，普通游线可按订单约定独立锁定。选择告知时，仍有效的相互竞争订单占用按可同时接受的数量核对。车组和点位容量为人数，同车同日的上限在多个产品同时引用时仍只有一份。

## `notices.json`

一批可能窗口只有一个实际兑现结果。

| 字段 | 含义 |
| --- | --- |
| `event` | 观礼批次 |
| `notice` | 通告次序 |
| `kind` | 预告、调整或确认 |
| `known` | 通告日期 |
| `possible_days` | 届时仍可能的实际观礼日 |
| `site` | 公布片区 |

## `parties.json`

人数按订单一同安置，旅客只能接受其可接受的日期和产品。

| 字段 | 含义 |
| --- | --- |
| `booking` | 订单号 |
| `event` | 观礼批次 |
| `first_day` | 原售日期 |
| `tour` | 同期销售组 |
| `people` | 需要一起安排的人数 |
| `allowed_days` | 可接受日期 |
| `fallback` | 可接受的普通游线 |
| `refund` | 合同退费元每人 |
| `priority` | 较小者优先 |
| `until` | 回复截止日 |
| `existing` | 为继续有效的订单 |
| `option` | 保留方式 |

## `choices.json`

实际回复与已结算的成本，缺件仍未知。

| 字段 | 含义 |
| --- | --- |
| `booking` | 订单号 |
| `day` | 回复日 |
| `offer_day` | 回复对应的备选日期 |
| `choice` | 实际回复 |
| `actual_per_unit` | 已确认人均改期或退款元 |

## `allocations.json`

每个观礼批次内同日同车组、同接待区用同一上限。

| 字段 | 含义 |
| --- | --- |
| `event` | 观礼批次 |
| `day` | 备选服务日 |
| `product` | 备选行程 |
| `vehicle` | 车组 |
| `venue` | 接待区 |
| `car_seats` | 车组人数上限 |
| `site_seats` | 接待区人数上限 |
| `car_cost` | 换车成本元每人 |
| `site_cost` | 额外点位成本元每人 |

## `offers.json`

已经向旅客说出的备选单。

| 字段 | 含义 |
| --- | --- |
| `offer` | 告知单号 |
| `booking` | 订单号 |
| `option_day` | 备选日期 |
| `product` | 备选产品 |
| `state` | 已发未答、已拒绝或已接受 |
| `sent` | 告知日 |

## `terms.json`

新条款候选及比对预算。

| 字段 | 含义 |
| --- | --- |
| `terms` | 条款版本 |
| `deadline_days` | 最晚接受的原定日后天数 |
| `max_refund` | 本批退费准备金元 |
| `services` | 条款可提供产品 |
| `min_notice` | 最少通知提前日 |
