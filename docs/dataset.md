# Dataset

## What we use

**IEEE-CIS Fraud Detection** (Kaggle) — around 590,000 real e-commerce transactions, each labeled fraud or not fraud, with fields like card number, billing address, email domain, device info, and ~340 anonymized behavioral features contributed by Vesta (the payment processor behind the dataset).

We chose it because it's the most-benchmarked fraud dataset in current academic literature — including recent graph/network-based fraud research — which makes the approach easy to defend and compare against known results.

## What we found when we looked closely (EDA)

Before building anything, we checked the data rather than assuming it would behave as expected:

| Question | Answer |
|---|---|
| How rare is fraud? | **3.5%** of transactions (20,663 out of 590,540) — confirms this is a heavily imbalanced problem |
| How many transactions have device/identity info at all? | Only **23.8%** |
| How much device data is missing, for those that do? | `DeviceInfo` 80% missing, `DeviceType` 76% missing |
| How populated are the fields we plan to build the identity graph from? | `card1` 100%, `card2–6` 98–99.7%, `addr1/addr2` 88.9%, `P_emaildomain` 84.0% |
| How much time does the data span? | ~182 days (6 months) |

## The device-data caveat

Kaggle deliberately masked and stripped much of the raw device fingerprinting data before releasing this competition dataset (for privacy). The result: device fields are too sparse to be relied on as a primary signal — 76–88% missing depending on the field.

This isn't a blind spot we discovered by accident — it's why the identity graph (see [Classifier](classifier.md) and the upcoming graph docs) is built primarily from **card number, address, and email domain**, which are 84–100% populated, with device info used only as a secondary bonus signal when it happens to be present.

## Fields used, by purpose

| Purpose | Fields |
|---|---|
| Classifier input | `TransactionAmt`, `ProductCD`, card/address/email fields, `C1–C14`, `D1–D15`, `V1–V339` (anonymized Vesta features) |
| Identity graph edges | `card1–card6`, `addr1/addr2`, `P_emaildomain` |
| Behavioral-similarity graph edges | `C1–C14` (identity-count features), `D1–D15` (time-delta/velocity features), `TransactionAmt`, `ProductCD` — no new data needed, these already exist in the dataset |
| Secondary/bonus signal only | `DeviceType`, `DeviceInfo`, `id_01–id_38` |
