# 量化團隊實盤審核報告

這是一份對本量化專案進行深度審核的報告。我們從**「準備要投入真金白銀實盤交易」**的最嚴格標準出發，進行了全面審視。雖然在軟體工程層面（分層架構、測試覆蓋）做得不錯，但**在「量化金融邏輯與策略可行性」上充滿了毀滅性的紅旗（Red Flags）**。如果直接拿這套系統上實盤，絕對會面臨嚴重的虧損與回撤。

## 🚨 核心嚴重問題審視 (Critical Flaws for Live Trading)

### 1. 回測欺騙：以「合成數據」作為驗證標準 (Validation on Synthetic Data)
我們檢查了 `docs/strategy-report.md` 以及測試代碼 `tests/test_oos_backtest.py`、`tests/test_validation.py`，發現**所有的 Sharpe Ratio、Max Drawdown 等績效數據，竟然全是跑在 `synthetic_data.py` 產生的「合成/人造資料」上！**
* **致命點**：在量化開發中，合成數據只適合用來跑 Unit Test，**絕對不能**用來驗證策略邏輯（Alpha）。用隨機漫步或自建週期的人造資料算出的 Sharpe 實盤中參考價值是 **0**。這意味著所有策略實際上都處於「從未經過真實市場歷史資料檢驗」的裸奔狀態。

### 2. 回測與實盤的嚴重「不一致性」 (Backtest-to-Live Divergence)
在 `src/quantbot/strategy/base.py` 中，`AllocatorStrategyAdapter` 存在一個致命的設計缺陷：
```python
# src/quantbot/strategy/base.py
bar = OhlcvBar(
    ...
    volume=Decimal("0"),  # Volume not available from snapshots
)
```
* **致命點**：實盤執行時，系統依賴本地接收 `MarketSnapshot` 的時間戳來自行拼湊 K 線，並且**強制將 Volume 設為 0**。但在回測端，策略卻使用了正常的 OHLCV 假資料。這會導致：
    1. 像 `microstructure_flow` 這種依賴 VWAP、OBV 的策略，在實盤會完全失效。
    2. 回測的市場衝擊模型（Market Impact Model）因為 Volume=0，預測變成 0（無滑點假象）。這種巨大的回測與實盤不對齊（Data Leakage / Divergence）是量化大忌。

### 3. 虛假繁榮與過度擬合 (Failed Acceptance Criteria & Overfitting)
`docs/strategy-report.md` 裡訂出了很高的標準（Sharpe ≥ 1.5, Max DD ≤ 25%）。然而，我們加入了嚴格的摩擦成本（Taker fee = 5 bps, Slippage = 2 bps），重新運行了 3 年 OOS 回測腳本。
* **致命點**：結果發現**即便在「合成數據」上，也沒有任何一個策略達標**。此外，策略中寫死了大量參數（如 EMA=5/20, Vol Target=15%），未見任何 Walk-Forward 的參數平原檢定，過度擬合（Over-fitting）的嫌疑極大，換個市場環境就會失效。

---

## 📊 各策略 3 年 OOS 回測數據與實戰點評

我們執行了 `test_oos_backtest.py`，採用嚴格摩擦成本設定，以下是所有策略的真實輸出數據與點評：

| 策略 (Strategy) | CAGR | MaxDD | Sharpe | Sortino | WinRate | PF | Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **trend_following** | 17.17% | 59.93% | 0.55 | 1.02 | 25.96% | 1.66 | 1098 |
| **mean_reversion_markov** | 10.85% | 37.05% | 0.46 | 0.78 | 46.74% | 1.20 | 1057 |
| **adaptive_momentum** | 3.17% | 50.23% | 0.35 | 0.63 | 49.23% | 1.36 | 1099 |
| **vol_mean_reversion** | 0.23% | 29.02% | 0.12 | 0.21 | 48.65% | 1.17 | 1073 |
| **regime_switching** | -0.14% | 35.55% | 0.11 | 0.17 | 47.97% | 1.20 | 1059 |
| **ensemble** | -3.12% | 48.33% | 0.16 | 0.27 | 19.03% | 1.43 | 1098 |
| **cross_sectional_arb** | -5.26% | 28.54% | -0.28 | -0.51 | 44.66% | 1.13 | 1059 |
| **microstructure_flow** | -8.56% | 45.02% | -0.24 | -0.35 | 48.67% | 1.04 | 1050 |

### 🔪 逐一策略犀利點評：

1. **Trend Following (趨勢跟蹤)**
   * **點評**：最大回撤逼近 **60%**。加密貨幣市場經常出現長達數月的震盪洗盤，傳統 EMA 交叉策略勝率僅 25% 是正常的，但在高達 60% 回撤的過程中，實盤的 Kill Switch（設在 15%）早就被觸發，策略會立刻死翹翹，活不到迎來大趨勢的時候。

2. **Mean Reversion Markov (馬可夫均值回歸)**
   * **點評**：表現相對最好，但 Max DD 依舊高達 37%。馬可夫狀態切換模型（HMM）通常有嚴重的「滯後性（Lagging）」。當它偵測到「崩盤/高波動」狀態時，價格往往已經跌完了。此時才降倉位，只會錯過後續的反彈。

3. **Adaptive Momentum (自適應動能)**
   * **點評**：50% 回撤，利潤幾乎被吃光。純時間序列動能在高摩擦（手續費 + 滑點）的短線環境下，換倉成本過高，導致白忙一場。

4. **Vol Mean Reversion (波動率均值回歸)**
   * **點評**：年化報酬率只有 0.23%。基於 Garman-Klass 波動率做指標，剔除雜訊能力幾乎無效，頻繁進出只為了賺取極微小的波動均值，全給交易所打工了。

5. **Regime Switching (環境切換)** & **Ensemble (綜合動能趨勢)**
   * **點評**：Ensemble 的勝率僅有悲慘的 19%，報酬率為負（-3.12%）。這是典型的「**多重條件過度過濾（Over-fitting/Over-filtering）**」。要求 EMA交叉 + 絕對動能 + 趨勢強度三者同時吻合才進場，導致訊號發生時「趨勢已經走到末端」，一進場就接刀被停損出局。

6. **Cross Sectional Arb (截面套利/輪動)**
   * **點評**：CAGR -5.26%。用多因子對永續合約做截面打分並做等權重 Top-N 輪動。在幣圈，如果不深入考慮精細的資金費率（Funding Cashflow）做基差套利，單純輪動只會瘋狂產生雙邊換倉手續費。

7. **Microstructure Flow (微觀結構流)**
   * **點評**：最糟糕的策略（-8.56%）。命名為微觀結構，卻只用 OHLCV K線來跑。真正的微觀流動性策略需要 L2/L3 Order Book 深度資料與 Tick-level Trade 流向。用粗糙的 K 線做微觀策略，就像用望遠鏡看細菌，瞎子摸象。

---

## 💡 資深量化團隊的下一步整改建議 (Actionable Next Steps)

1. **廢棄合成數據驗證**：立刻停止所有基於 `synthetic_data.py` 的策略評估。下載 Binance / OKX 過去 3 到 5 年的「真實 1 分鐘 K 線」與「真實資金費率歷史」，重新面對現實。
2. **重構實盤 K 線接收邏輯**：拔除 `AllocatorStrategyAdapter` 裡自己用 Snapshot 拼 K 線的作法。改為透過 OKX WebSocket 直接訂閱官方的 `candle1m` 或 `candle5m` 頻道。確保實盤與回測吃到的 OHLCV 資料（尤其是 Volume）是 100% 一致的。
3. **策略斷捨離**：將 8 個複雜且不賺錢的策略全部封存。回到最基礎的 `trend_following` 或單一資金費率套利（Funding Arbitrage）。直到能在「真實歷史數據」且「包含真實手續費與市場衝擊」的回測中做到 Sharpe > 1.2，再來談馬可夫模型跟截面輪動。
