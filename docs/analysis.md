# Model Performance Summary

# AUC

---

## Classification Metrics
| Metric        | XGBoost | LightGBM |
|---------------|---------|----------|
| Accuracy      | **0.57** | 0.56 |
| Precision     | 0.57 | 0.57 |
| Recall        | **0.55** | 0.53 |
| F1 Score      | **0.56** | 0.55 |
| Logloss       | 0.75 | **0.69** |
| AUC           | **0.60** | 0.59 |

---

## Trading Performance
| Metric             | XGBoost   | LightGBM   |
|--------------------|-----------|------------|
| Total Trades       | **29,898** | 21,802 |
| Total PnL          | 572,998 | **616,046** |
| Avg PnL / Trade    | 19.17 | **28.26** |
| Win Rate (Total)   | **49.4%** | 36.8% |
| Long Win Rate      | 23.8% | 16.6% |
| Short Win Rate     | 25.6% | 20.3% |
| Sharpe Ratio       | 1.32 | **1.52** |
| Max Drawdown       | 29,697 | **23,929** |

---

## PnL by Hour of Day
| Hour | XGBoost PnL | LightGBM PnL |
|------|-------------|--------------|
| 0    | 21,743 | 11,778 |
| 1    | 5,644  | 8,376 |
| 2    | 7,453  | 8,535 |
| 3    | 8,849  | 11,857 |
| 4    | 7,584  | 2,696 |
| 5    | 29,367 | 11,121 |
| 6    | 36,507 | 30,691 |
| 7    | 38,834 | 38,978 |
| 8    | 39,025 | 46,189 |
| 9    | 38,308 | 51,424 |
| 10   | 79,840 | 87,373 |
| 11   | 41,194 | 59,230 |
| 12   | 49,850 | 51,704 |
| 13   | 35,013 | 36,380 |
| 14   | 42,416 | 32,029 |
| 15   | 24,129 | 20,219 |
| 16   | 4,509  | -6,545 |
| 17   | -1,191 | 28,104 |
| 18   | -5,978 | 30,606 |
| 19   | 26,996 | 16,960 |
| 20   | 22,785 | 12,165 |
| 21   | 9,139  | 2,412 |
| 22   | 13,066 | 7,208 |
| 23   | -2,085 | 16,556 |

---

## PnL by Day of Week
| Day | XGBoost PnL | LightGBM PnL |
|-----|-------------|--------------|
| Mon | 125,098 | 130,792 |
| Tue | 118,699 | 101,937 |
| Wed | 62,632  | 56,177 |
| Thu | 107,050 | 107,107 |
| Fri | 52,871  | 61,572 |
| Sat | 71,456  | 50,693 |
| Sun | 35,193  | 107,769 |

---

## PnL by Month
| Month | XGBoost PnL | LightGBM PnL |
|-------|-------------|--------------|
| Jan   | 18,091 | 25,830 |
| Feb   | 60,340 | 58,753 |
| Mar   | 40,293 | 45,876 |
| Apr   | 149,171 | 129,406 |
| May   | 55,678 | 52,436 |
| Jun   | 15,424 | 62,403 |
| Jul   | 56,228 | 57,519 |
| Aug   | 47,163 | 71,830 |
| Sep   | 62,212 | 53,801 |
| Oct   | 12,938 | 7,236 |
| Nov   | 27,061 | 18,274 |
| Dec   | 28,401 | 32,681 |

---

## PnL by Quarter
| Quarter | XGBoost PnL | LightGBM PnL |
|---------|-------------|--------------|
| Q1      | 118,723 | 130,459 |
| Q2      | 220,273 | 244,246 |
| Q3      | 165,602 | 183,151 |
| Q4      | 68,400  | 58,191 |

---

## Weekends & Holidays
| Category  | XGBoost PnL | LightGBM PnL |
|-----------|-------------|--------------|
| Weekdays  | 466,349 | 457,584 |
| Weekends  | 106,650 | 158,462 |
| Non-Hol.  | 528,933 | 570,801 |
| Holidays  | 44,065  | 45,245 |

---

## Key Takeaways
- **XGBoost** → More trades, higher win rate, but lower efficiency (Sharpe) and higher drawdowns.  
- **LightGBM** → Fewer trades, lower win rate, but **higher PnL, Sharpe, PnL/trade, and lower drawdowns**.  
- Both models peak in **Q2** and are strongest during mid-morning hours (9–12h).  
- Weekends are more profitable under LightGBM, while weekdays dominate for XGBoost.  
- Holidays are profitable but less significant than normal days.  
